import pickle, numpy as np, os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

import os
from utils.repo_paths import col_pywork, results_dir

TRACKS = os.path.join(col_pywork(), "tracks.pckl")


def _scores(workdir, filename="scores.pckl"):
    return os.path.join(col_pywork(workdir), filename)


score_files = {
    "LR-ASD": _scores("pywork_baseline"),
    "Transformer": _scores("pywork", "scores_ablation1.pckl"),
    "Multi-face": _scores("pywork_multiface"),
    "CIR-020": _scores("pywork_cir020"),
    "TalkNCE": _scores("pywork_talknce"),
}

import glob
from sklearn.metrics import f1_score
from collections import defaultdict

ORIG_FPS = 29.97; OUR_FPS = 25.0; RATIO = OUR_FPS/ORIG_FPS
vidTracks = pickle.load(open(TRACKS,'rb'))
flist = sorted(glob.glob(f'{COLDATA}/col/pyframes/*.jpg'))
first_frame = int(os.path.splitext(os.path.basename(flist[0]))[0])
total_frames = len(flist)
names = ['bell','boll','lieb','long','sick']

gt_per_speaker = {n:{} for n in names}
for name in names:
    with open(f'{COLDATA}/col_labels/fusion/{name}.txt') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts)<5: continue
            gt_frame = int(int(parts[0])*RATIO)
            if gt_frame <= total_frames:
                gt_per_speaker[name][gt_frame] = int(parts[4])

def compute_ece(scores_flat, labels_flat, n_bins=10):
    """Expected Calibration Error"""
    # Convert raw scores to probabilities using sigmoid-like normalization
    scores_arr = np.array(scores_flat)
    # Normalize to [0,1] using min-max
    s_min, s_max = scores_arr.min(), scores_arr.max()
    probs = (scores_arr - s_min) / (s_max - s_min + 1e-8)
    
    labels_arr = np.array(labels_flat)
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if mask.sum() == 0: continue
        avg_conf = probs[mask].mean()
        avg_acc  = labels_arr[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)
    return ece

gt_frame_sets = {n: set(gt_per_speaker[n].keys()) for n in names}

def get_frame_scores_labels(scores):
    track_data = {}
    for tidx, track in enumerate(vidTracks):
        if tidx >= len(scores): continue
        score = scores[tidx]
        fr, sc = [], []
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            if frame >= total_frames: continue
            s = float(np.mean(score[max(fidx-2,0):min(fidx+3,len(score)-1)]))
            fr.append(frame + first_frame)
            sc.append(s)
        if len(fr) > 50:
            track_data[tidx] = (set(fr), fr, sc)

    all_assignments = []
    for tidx, (frame_set, fr, sc) in track_data.items():
        for name in names:
            overlap = len(frame_set & gt_frame_sets[name])
            norm = overlap / max(len(gt_frame_sets[name]),1)
            if overlap > 100:
                all_assignments.append((norm, overlap, tidx, name))
    all_assignments.sort(reverse=True)
    track_to_speaker = {}
    for norm, overlap, tidx, name in all_assignments:
        if tidx not in track_to_speaker:
            track_to_speaker[tidx] = (name, overlap, norm)

    all_scores, all_labels = [], []
    for tidx, (speaker, _, _) in track_to_speaker.items():
        _, fr, sc = track_data[tidx]
        for frame, s in zip(fr, sc):
            if frame in gt_per_speaker[speaker]:
                all_scores.append(s)
                all_labels.append(gt_per_speaker[speaker][frame])
    return all_scores, all_labels

print(f"\n{'Model':<15} {'ECE':>8} {'N_frames':>10}")
print("-" * 40)
ece_results = {}
for mname, path in score_files.items():
    if not os.path.exists(path):
        print(f"{mname:<15} MISSING")
        continue
    scores = pickle.load(open(path,'rb'))
    s, l = get_frame_scores_labels(scores)
    ece = compute_ece(s, l)
    ece_results[mname] = ece
    print(f"{mname:<15} {ece:>8.4f} {len(s):>10}")

# Plot reliability diagrams
fig, axes = plt.subplots(1, len(ece_results), figsize=(4*len(ece_results), 4))
if len(ece_results) == 1: axes = [axes]

for ax, (mname, _) in zip(axes, ece_results.items()):
    path = score_files[mname]
    scores = pickle.load(open(path,'rb'))
    s, l = get_frame_scores_labels(scores)
    scores_arr = np.array(s)
    probs = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-8)
    labels_arr = np.array(l)
    
    bin_boundaries = np.linspace(0, 1, 11)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    accuracies = []
    for i in range(10):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if mask.sum() == 0:
            accuracies.append(0)
        else:
            accuracies.append(labels_arr[mask].mean())
    
    ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect calibration')
    ax.bar(bin_centers, accuracies, width=0.1, alpha=0.7, color='steelblue', label='Model')
    ax.set_xlabel('Confidence', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title(f'{mname}\nECE={ece_results[mname]:.3f}', fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle('Reliability Diagrams — Columbia Cross-Domain', fontsize=12, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(results_dir(), "uncertainty_ece.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
