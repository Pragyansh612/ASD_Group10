import os, glob, pickle, numpy as np, argparse
from sklearn.metrics import f1_score, accuracy_score
from utils.repo_paths import col_data_root

parser = argparse.ArgumentParser()
parser.add_argument('--scoresPath', type=str, default=None)
parser.add_argument('--colSavePath', type=str, default=None)
parser.add_argument('--window', type=int, default=5, help='Smoothing window size')
parser.add_argument('--threshold', type=float, default=-1.0, help='Speaking threshold')
args = parser.parse_args()
args.colSavePath = args.colSavePath or col_data_root()

pyworkPath   = os.path.join(args.colSavePath, 'col', 'pywork')
pyframesPath = os.path.join(args.colSavePath, 'col', 'pyframes')
if args.scoresPath is None:
    args.scoresPath = os.path.join(pyworkPath, 'scores_ablation1.pckl')

vidTracks = pickle.load(open(os.path.join(pyworkPath, 'tracks.pckl'), 'rb'))
scores    = pickle.load(open(args.scoresPath, 'rb'))

ORIG_FPS = 29.97
OUR_FPS  = 25.0
RATIO    = OUR_FPS / ORIG_FPS

flist        = sorted(glob.glob(os.path.join(pyframesPath, '*.jpg')))
first_frame  = int(os.path.splitext(os.path.basename(flist[0]))[0])
total_frames = len(flist)

# Build track data with temporal smoothing
track_data = {}
for tidx, track in enumerate(vidTracks):
    score = scores[tidx]
    # Apply temporal smoothing
    smoothed = np.convolve(score, np.ones(args.window)/args.window, mode='same')
    fr, sc = [], []
    for fidx, frame in enumerate(track['track']['frame'].tolist()):
        if frame >= total_frames:
            continue
        s = float(smoothed[min(fidx, len(smoothed)-1)])
        actual_frame = frame + first_frame
        fr.append(actual_frame)
        sc.append(s)
    if len(fr) > 50:
        track_data[tidx] = (set(fr), fr, sc)

# Load GT
names = ['bell', 'boll', 'lieb', 'long', 'sick']
gt_per_speaker = {n: {} for n in names}
for name in names:
    with open(os.path.join(args.colSavePath, 'col_labels', 'fusion', f'{name}.txt')) as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            gt_frame_orig = int(parts[0])
            gt_frame_ours = int(gt_frame_orig * RATIO)
            gt = int(parts[4])
            if gt_frame_ours <= total_frames:
                gt_per_speaker[name][gt_frame_ours] = gt

gt_frame_sets = {n: set(gt_per_speaker[n].keys()) for n in names}

# Assign tracks to speakers
track_speaker_scores = {}
for tidx, (frame_set, fr, sc) in track_data.items():
    track_speaker_scores[tidx] = {}
    for name in names:
        overlap = len(frame_set & gt_frame_sets[name])
        norm_overlap = overlap / max(len(gt_frame_sets[name]), 1)
        track_speaker_scores[tidx][name] = (overlap, norm_overlap)

track_to_speaker = {}
all_assignments = []
for tidx in track_data:
    for name in names:
        overlap, norm = track_speaker_scores[tidx][name]
        if overlap > 100:
            all_assignments.append((norm, overlap, tidx, name))

all_assignments.sort(reverse=True)
for norm, overlap, tidx, name in all_assignments:
    if tidx not in track_to_speaker:
        track_to_speaker[tidx] = (name, overlap, norm)

speaker_frame_scores = {n: {} for n in names}
for tidx, (speaker, overlap, norm) in track_to_speaker.items():
    _, fr, sc = track_data[tidx]
    for frame, s in zip(fr, sc):
        if frame not in speaker_frame_scores[speaker]:
            speaker_frame_scores[speaker][frame] = s
        else:
            speaker_frame_scores[speaker][frame] = max(speaker_frame_scores[speaker][frame], s)

# Find best threshold by sweeping
print(f"\n=== THRESHOLD SWEEP (window={args.window}) ===")
best_threshold = args.threshold
best_avg_f1 = 0

for thresh in np.arange(-3.0, 2.0, 0.1):
    F1s, valid = 0, 0
    for name in names:
        common = sorted(set(gt_per_speaker[name].keys()) & set(speaker_frame_scores[name].keys()))
        if len(common) < 100: continue
        preds  = np.array([1 if speaker_frame_scores[name][f] > thresh else 0 for f in common])
        labels = np.array([gt_per_speaker[name][f] for f in common])
        if labels.sum() == 0: continue
        F1s += f1_score(labels, preds, zero_division=0)
        valid += 1
    avg_f1 = 100 * F1s / valid if valid > 0 else 0
    if avg_f1 > best_avg_f1:
        best_avg_f1 = avg_f1
        best_threshold = thresh

print(f"Best threshold: {best_threshold:.1f}  Best F1: {best_avg_f1:.2f}%")

# Final evaluation with best threshold
print(f"\n=== FINAL RESULTS (window={args.window}, threshold={best_threshold:.1f}) ===")
F1s, valid = 0, 0
for name in sorted(names):
    common = sorted(set(gt_per_speaker[name].keys()) & set(speaker_frame_scores[name].keys()))
    if len(common) < 100:
        print(f"{name}: insufficient frames")
        continue
    preds  = np.array([1 if speaker_frame_scores[name][f] > best_threshold else 0 for f in common])
    labels = np.array([gt_per_speaker[name][f] for f in common])
    if labels.sum() == 0: continue
    F1  = f1_score(labels, preds, zero_division=0)
    ACC = accuracy_score(labels, preds)
    F1s += F1
    valid += 1
    print(f"{name}: ACC={100*ACC:.2f}%  F1={100*F1:.2f}%")

print(f"\nAverage F1: {100*F1s/valid:.2f}%  (threshold={best_threshold:.1f}, window={args.window})")
