import pickle, numpy as np, os, glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

COLDATA = '/usershome/cs671_user6/asd_project/ColData'
PYFRAMES = f'{COLDATA}/col/pyframes'
TRACKS   = f'{COLDATA}/col/pywork/tracks.pckl'

score_files = {
    'LR-ASD':        f'{COLDATA}/col/pywork_baseline/scores.pckl',
    'Transformer':   f'{COLDATA}/col/pywork/scores_ablation1.pckl',
    'Multi-face':    f'{COLDATA}/col/pywork_multiface/scores.pckl',
    'Large':         f'{COLDATA}/col/pywork_ablation3/scores.pckl',
    'Augmentation':  f'{COLDATA}/col/pywork_improved/scores.pckl',
    'Hard-neg':      f'{COLDATA}/col/pywork_hardneg/scores.pckl',
    'Attn-ctx':      f'{COLDATA}/col/pywork_attn_context/scores.pckl',
    'TalkNCE':       f'{COLDATA}/col/pywork_talknce/scores.pckl',
    'CIR-020':       f'{COLDATA}/col/pywork_cir020/scores.pckl',
}

from sklearn.metrics import f1_score, accuracy_score

ORIG_FPS = 29.97
OUR_FPS  = 25.0
RATIO    = OUR_FPS / ORIG_FPS

vidTracks = pickle.load(open(TRACKS, 'rb'))
flist     = sorted(glob.glob(f'{PYFRAMES}/*.jpg'))
first_frame  = int(os.path.splitext(os.path.basename(flist[0]))[0])
total_frames = len(flist)

names = ['bell', 'boll', 'lieb', 'long', 'sick']

# Load GT
gt_per_speaker = {n: {} for n in names}
for name in names:
    with open(f'{COLDATA}/col_labels/fusion/{name}.txt') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            gt_frame = int(int(parts[0]) * RATIO)
            gt = int(parts[4])
            if gt_frame <= total_frames:
                gt_per_speaker[name][gt_frame] = gt

gt_frame_sets = {n: set(gt_per_speaker[n].keys()) for n in names}

def eval_model_scores(scores):
    track_data = {}
    for tidx, track in enumerate(vidTracks):
        if tidx >= len(scores): continue
        score = scores[tidx]
        fr, sc = [], []
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            if frame >= total_frames: continue
            s = float(np.mean(score[max(fidx-2,0):min(fidx+3,len(score)-1)]))
            actual_frame = frame + first_frame
            fr.append(actual_frame)
            sc.append(s)
        if len(fr) > 50:
            track_data[tidx] = (set(fr), fr, sc)

    # Assign tracks to speakers
    all_assignments = []
    for tidx, (frame_set, fr, sc) in track_data.items():
        for name in names:
            overlap = len(frame_set & gt_frame_sets[name])
            norm = overlap / max(len(gt_frame_sets[name]), 1)
            if overlap > 100:
                all_assignments.append((norm, overlap, tidx, name))

    all_assignments.sort(reverse=True)
    track_to_speaker = {}
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

    f1s = {}
    for name in names:
        common = sorted(set(gt_per_speaker[name].keys()) & set(speaker_frame_scores[name].keys()))
        if len(common) < 100: 
            f1s[name] = 0.0
            continue
        preds  = np.array([1 if speaker_frame_scores[name][f] > -1 else 0 for f in common])
        labels = np.array([gt_per_speaker[name][f] for f in common])
        if labels.sum() == 0:
            f1s[name] = 0.0
            continue
        f1s[name] = 100 * f1_score(labels, preds, zero_division=0)
    return f1s

# Build matrix
model_names = list(score_files.keys())
matrix = np.zeros((len(model_names), len(names)))

for i, (mname, path) in enumerate(score_files.items()):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    scores = pickle.load(open(path, 'rb'))
    f1s = eval_model_scores(scores)
    for j, name in enumerate(names):
        matrix[i, j] = f1s[name]
    print(f"{mname}: {f1s}")

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.capitalize() for n in names], fontsize=12)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=11)

for i in range(len(model_names)):
    for j in range(len(names)):
        ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if matrix[i,j] < 40 or matrix[i,j] > 80 else 'black')

plt.colorbar(im, ax=ax, label='F1 Score (%)')
ax.set_title('Per-Speaker Columbia F1 by Model\n(Green=high F1, Red=low F1)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/usershome/cs671_user6/asd_project/LR-ASD/speaker_heatmap.png',
            dpi=150, bbox_inches='tight')
print("Saved speaker_heatmap.png")
