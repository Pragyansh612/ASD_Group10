import os, pickle, numpy as np, glob, argparse, csv
from scipy.stats import pearsonr
from collections import defaultdict
from utils.repo_paths import col_pywork, results_dir

parser = argparse.ArgumentParser(description="Inter-face score correlation on Columbia")
parser.add_argument("--colSavePath", type=str, default=None, help="Unused; kept for CLI compatibility")
parser.add_argument("--tracksWorkdir", type=str, default="pywork", help="Workdir name under ColData/col/")
args = parser.parse_args()

tracks_path = os.path.join(col_pywork(args.tracksWorkdir), "tracks.pckl")

def _scores(workdir, filename="scores.pckl"):
    return os.path.join(col_pywork(workdir), filename)


score_files = {
    "LR-ASD_baseline": _scores("pywork"),
    "Transformer": _scores("pywork", "scores_ablation1.pckl"),
    "Multi-face": _scores("pywork_ablation3"),
    "Large_capacity": _scores("pywork_ablation3"),
    "Augmentation": _scores("pywork_improved"),
    "Hard_negatives": _scores("pywork_hardneg"),
    "Attn_context": _scores("pywork_attn_context"),
    "TalkNet": _scores("pywork_talknet"),
}

print("Checking available score files:")
available = {}
for name, path in score_files.items():
    if os.path.exists(path):
        print(f"  {name}: {path} ✓")
        available[name] = path
    else:
        print(f"  {name}: MISSING (run run_scoring.py for this variant first)")

if not os.path.exists(tracks_path):
    raise FileNotFoundError(f"tracks.pckl not found: {tracks_path}")

vidTracks = pickle.load(open(tracks_path, "rb"))
print(f"\nLoaded {len(vidTracks)} tracks")


def compute_correlation(vidTracks, scores):
    frame_to_faces = defaultdict(list)

    for tidx, track in enumerate(vidTracks):
        if tidx >= len(scores):
            continue
        score = scores[tidx]
        frames = track["track"]["frame"].tolist()
        for fidx, frame in enumerate(frames):
            if fidx < len(score):
                s = float(score[fidx])
                frame_to_faces[frame].append((tidx, s))

    pairs_A, pairs_B = [], []
    for frame, faces in frame_to_faces.items():
        if len(faces) == 2:
            pairs_A.append(faces[0][1])
            pairs_B.append(faces[1][1])

    print(f"  Frames with exactly 2 faces: {len(pairs_A)}")

    if len(pairs_A) < 10:
        return 0.0, 0

    corr, _ = pearsonr(pairs_A, pairs_B)
    return corr, len(pairs_A)


print("\n=== CO-OCCURRENCE CORRELATION ANALYSIS ===")
print(f"{'Model':<20} {'Correlation':>12} {'N_frames':>10} {'Interpretation'}")
print("-" * 65)

results = []
for name, path in available.items():
    scores = pickle.load(open(path, "rb"))
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

out_csv = os.path.join(results_dir(), "cooccurrence_results.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "columbia_correlation", "n_frames"])
    for name, corr, n in results:
        writer.writerow([name, f"{corr:.4f}", n])

print(f"\nSaved to {out_csv}")
