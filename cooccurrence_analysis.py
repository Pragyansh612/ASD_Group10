import os, pickle, numpy as np, glob
from scipy.stats import pearsonr
from collections import defaultdict

# All models and their scores.pckl paths
MODELS = {
    'TalkNet':        '/usershome/cs671_user6/asd_project/ColData/col/pywork/scores_ablation1.pckl',  # reuse transformer path as placeholder
    'LR-ASD':         '/usershome/cs671_user6/asd_project/ColData/col/pywork/scores.pckl',
    'Transformer':    '/usershome/cs671_user6/asd_project/ColData/col/pywork/scores_ablation1.pckl',
    'Multi-face':     '/usershome/cs671_user6/asd_project/ColData/col/pywork_ablation3/scores.pckl',
    'Large':          '/usershome/cs671_user6/asd_project/ColData/col/pywork_ablation3/scores.pckl',
    'Augmentation':   '/usershome/cs671_user6/asd_project/ColData/col/pywork_improved/scores.pckl',
    'HardNeg':        '/usershome/cs671_user6/asd_project/ColData/col/pywork_hardneg/scores.pckl',
    'AttnContext':    '/usershome/cs671_user6/asd_project/ColData/col/pywork_attn_context/scores.pckl',
}

# We need separate scores for each model - let's check what we actually have
print("Checking available score files:")
available = {}
score_files = {
    'LR-ASD_baseline':  '/usershome/cs671_user6/asd_project/ColData/col/pywork/scores.pckl',
    'Transformer':      '/usershome/cs671_user6/asd_project/ColData/col/pywork/scores_ablation1.pckl',
    'Multi-face':       '/usershome/cs671_user6/asd_project/ColData/col/pywork_ablation3/scores.pckl',
    'Large_capacity':   '/usershome/cs671_user6/asd_project/ColData/col/pywork_ablation3/scores.pckl',
    'Augmentation':     '/usershome/cs671_user6/asd_project/ColData/col/pywork_improved/scores.pckl',
    'Hard_negatives':   '/usershome/cs671_user6/asd_project/ColData/col/pywork_hardneg/scores.pckl',
    'Attn_context':     '/usershome/cs671_user6/asd_project/ColData/col/pywork_attn_context/scores.pckl',
}

for name, path in score_files.items():
    if os.path.exists(path):
        print(f"  {name}: {path} ✓")
        available[name] = path
    else:
        print(f"  {name}: MISSING")

# Load tracks
tracks_path = '/usershome/cs671_user6/asd_project/ColData/col/pywork/tracks.pckl'
vidTracks = pickle.load(open(tracks_path, 'rb'))
print(f"\nLoaded {len(vidTracks)} tracks")

def compute_correlation(vidTracks, scores):
    """
    For each frame with exactly 2 face tracks visible,
    record (score_A, score_B) and compute Pearson correlation.
    """
    # Build frame -> list of (track_idx, score) mapping
    frame_to_faces = defaultdict(list)
    
    for tidx, track in enumerate(vidTracks):
        if tidx >= len(scores):
            continue
        score = scores[tidx]
        frames = track['track']['frame'].tolist()
        for fidx, frame in enumerate(frames):
            if fidx < len(score):
                s = float(score[fidx])
                frame_to_faces[frame].append((tidx, s))
    
    # Find frames with exactly 2 faces
    pairs_A, pairs_B = [], []
    for frame, faces in frame_to_faces.items():
        if len(faces) == 2:
            pairs_A.append(faces[0][1])
            pairs_B.append(faces[1][1])
    
    print(f"  Frames with exactly 2 faces: {len(pairs_A)}")
    
    if len(pairs_A) < 10:
        return 0.0, 0
    
    corr, pval = pearsonr(pairs_A, pairs_B)
    return corr, len(pairs_A)

print("\n=== CO-OCCURRENCE CORRELATION ANALYSIS ===")
print(f"{'Model':<20} {'Correlation':>12} {'N_frames':>10} {'Interpretation'}")
print("-" * 65)

results = []
for name, path in available.items():
    scores = pickle.load(open(path, 'rb'))
    corr, n = compute_correlation(vidTracks, scores)
    
    if corr < -0.3:
        interp = "STRONG spurious (bad)"
    elif corr < -0.1:
        interp = "Moderate spurious"
    elif corr < 0.1:
        interp = "Near-zero (good)"
    else:
        interp = "Positive correlation"
    
    print(f"{name:<20} {corr:>12.4f} {n:>10} {interp}")
    results.append((name, corr, n))

# Save CSV
import csv
with open('/usershome/cs671_user6/asd_project/LR-ASD/cooccurrence_results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'columbia_correlation', 'n_frames'])
    for name, corr, n in results:
        writer.writerow([name, f"{corr:.4f}", n])

print("\nSaved to cooccurrence_results.csv")
print("\nKey insight: Multi-face models should show strong negative correlation")
print("(if A speaks, B predicted not to speak) — spurious AVA pattern")
print("Single-face/transformer models should show near-zero correlation")
