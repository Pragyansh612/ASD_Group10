import os, glob, pickle, numpy as np, argparse
from sklearn.metrics import f1_score, accuracy_score
from utils.repo_paths import col_data_root, col_pyframes, col_pywork, results_dir

parser = argparse.ArgumentParser(
    description="Columbia F1 evaluation (fps-corrected frame-overlap protocol)"
)
parser.add_argument("--pyworkPath", type=str, default=None, help="Dir with tracks.pckl and scores.pckl")
parser.add_argument("--pyframesPath", type=str, default=None, help="Extracted Columbia frames")
parser.add_argument("--colSavePath", type=str, default=None, help="ColData root (contains col_labels/)")
args = parser.parse_args()

pyworkPath   = args.pyworkPath or col_pywork()
pyframesPath = args.pyframesPath or col_pyframes()
colSavePath  = args.colSavePath or col_data_root()

# fps conversion factor
ORIG_FPS = 29.97
OUR_FPS  = 25.0
RATIO    = OUR_FPS / ORIG_FPS  # 0.8342

vidTracks = pickle.load(open(os.path.join(pyworkPath, 'tracks.pckl'), 'rb'))
scores    = pickle.load(open(os.path.join(pyworkPath, 'scores.pckl'), 'rb'))

flist        = sorted(glob.glob(os.path.join(pyframesPath, '*.jpg')))
first_frame  = int(os.path.splitext(os.path.basename(flist[0]))[0])
total_frames = len(flist)
print(f'Total frames: {total_frames}, first_frame: {first_frame}')

# build track data — track frames are in our 25fps space
track_data = {}
for tidx, track in enumerate(vidTracks):
    score  = scores[tidx]
    fr, sc = [], []
    for fidx, frame in enumerate(track['track']['frame'].tolist()):
        if frame >= total_frames:
            continue
        s = float(np.mean(score[max(fidx-2,0): min(fidx+3,len(score)-1)]))
        actual_frame = frame + first_frame  # our 25fps frame number
        fr.append(actual_frame)
        sc.append(s)
    if len(fr) > 50:
        track_data[tidx] = (set(fr), fr, sc)

print(f'Substantial tracks: {len(track_data)}')

# load GT — convert GT frame numbers from 29.97fps to 25fps
names = ['bell', 'boll', 'lieb', 'long', 'sick']
gt_per_speaker = {n: {} for n in names}

for name in names:
    with open(os.path.join(colSavePath, 'col_labels', 'fusion', f'{name}.txt')) as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            gt_frame_orig = int(parts[0])
            gt_frame_ours = int(gt_frame_orig * RATIO)  # convert to 25fps
            gt            = int(parts[4])
            if gt_frame_ours <= total_frames:
                gt_per_speaker[name][gt_frame_ours] = gt

print(f'GT frames after fps conversion:')
for name in names:
    speaking = sum(gt_per_speaker[name].values())
    print(f'  {name}: total={len(gt_per_speaker[name])}  speaking={speaking}')

gt_frame_sets = {n: set(gt_per_speaker[n].keys()) for n in names}

# assign tracks to speakers using normalized overlap
track_speaker_scores = {}
for tidx, (frame_set, fr, sc) in track_data.items():
    track_speaker_scores[tidx] = {}
    for name in names:
        overlap      = len(frame_set & gt_frame_sets[name])
        norm_overlap = overlap / max(len(gt_frame_sets[name]), 1)
        track_speaker_scores[tidx][name] = (overlap, norm_overlap)

# assign each track to best normalized speaker
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

# print assignments per speaker
print('\nTrack assignments:')
speaker_tracks = {n: [] for n in names}
for tidx, (spk, ov, norm) in track_to_speaker.items():
    speaker_tracks[spk].append((tidx, ov, norm))

for name in names:
    tracks = sorted(speaker_tracks[name], key=lambda x: x[1], reverse=True)
    total_frames_assigned = sum(len(track_data[t][1]) for t, _, _ in tracks)
    spk_rates = []
    for tidx, ov, norm in tracks[:3]:
        _, fr, sc = track_data[tidx]
        spk_rates.append(f'T{tidx}:{np.mean([1 if s>0 else 0 for s in sc]):.2f}')
    print(f'  {name}: {len(tracks)} tracks, {total_frames_assigned} frames, top3_spk={spk_rates}')

# build per-speaker prediction timeline
speaker_frame_scores = {n: {} for n in names}
for tidx, (speaker, overlap, norm) in track_to_speaker.items():
    _, fr, sc = track_data[tidx]
    for frame, s in zip(fr, sc):
        if frame not in speaker_frame_scores[speaker]:
            speaker_frame_scores[speaker][frame] = s
        else:
            speaker_frame_scores[speaker][frame] = max(
                speaker_frame_scores[speaker][frame], s
            )

# evaluate
print('\n=== COLUMBIA F1 RESULTS (fps-corrected) ===')
F1s   = 0
valid = 0
for name in sorted(names):
    common = sorted(
        set(gt_per_speaker[name].keys()) & set(speaker_frame_scores[name].keys())
    )
    if len(common) < 100:
        print(f'{name}: insufficient frames ({len(common)}) — skipping')
        continue
    preds  = np.array([1 if speaker_frame_scores[name][f] > -1 else 0 for f in common])
    labels = np.array([gt_per_speaker[name][f] for f in common])
    if labels.sum() == 0:
        print(f'{name}: no speaking frames in range — skipping')
        continue
    F1  = f1_score(labels, preds, zero_division=0)
    ACC = accuracy_score(labels, preds)
    F1s   += F1
    valid += 1
    print(f'{name}: ACC={100*ACC:.2f}%  F1={100*F1:.2f}%  frames={len(common)}  pred_spk={preds.sum()}  gt_spk={labels.sum()}')

avg_f1 = 100 * (F1s / valid) if valid > 0 else 0
print(f'\nOur Average F1: {avg_f1:.2f}%  (over {valid} speakers)')
print(f'Paper reports:  86.10%')

with open(os.path.join(results_dir(), 'summary.txt'), 'a') as f:
    f.write(f'\nColumbia F1 final (fps corrected): {avg_f1:.2f}%\n')
    f.write(f'Paper Columbia F1:                 86.10%\n')
    f.write(f'Note: evaluated on 4/5 speakers (lieb has no speaking frames in our video range)\n')
print('\nSaved to results/summary.txt')
