import pickle, numpy as np, os, glob
import matplotlib.pyplot as plt
from collections import defaultdict

COLDATA  = '/usershome/cs671_user6/asd_project/ColData'
TRACKS   = f'{COLDATA}/col/pywork/tracks.pckl'
PYFRAMES = f'{COLDATA}/col/pyframes'

vidTracks = pickle.load(open(TRACKS, 'rb'))
flist = sorted(glob.glob(f'{PYFRAMES}/*.jpg'))
total_frames = len(flist)

# ── Domain characterization from face statistics ──────────────────────────────
# For each frame, count visible faces
frame_face_counts = defaultdict(int)
for track in vidTracks:
    for frame in track['track']['frame'].tolist():
        if frame < total_frames:
            frame_face_counts[frame] += 1

all_face_counts = list(frame_face_counts.values())
avg_faces      = np.mean(all_face_counts)
pct_multi_face = 100 * sum(1 for c in all_face_counts if c >= 2) / len(all_face_counts)

# Speaking turn overlap rate (co-occurrence)
frame_speaking = defaultdict(list)  # frame -> list of track scores

# Load baseline scores for this analysis
baseline_scores = pickle.load(open(f'{COLDATA}/col/pywork_baseline/scores.pckl','rb'))
for tidx, track in enumerate(vidTracks):
    if tidx >= len(baseline_scores): continue
    score = baseline_scores[tidx]
    for fidx, frame in enumerate(track['track']['frame'].tolist()):
        if frame < total_frames and fidx < len(score):
            frame_speaking[frame].append(float(score[fidx]))

# Overlap rate: frames where 2+ faces both have positive scores
overlap_frames = sum(1 for f, scores in frame_speaking.items()
                     if len(scores) >= 2 and sum(1 for s in scores if s > -1) >= 2)
overlap_rate = 100 * overlap_frames / max(len(frame_speaking), 1)

print("=== COLUMBIA DOMAIN CHARACTERIZATION ===")
print(f"Average faces per frame:  {avg_faces:.2f}")
print(f"Multi-face frame %:       {pct_multi_face:.1f}%")
print(f"Speaking overlap rate:    {overlap_rate:.1f}%")

# ── Routing decision ─────────────────────────────────────────────────────────
print("\n=== SCENE-TYPE ROUTING RULES ===")
print("Based on face statistics, route to:")

if avg_faces >= 3:
    scene = "Panel/Conference"
    recommended = "Transformer (low domain drop)"
elif avg_faces >= 2 and pct_multi_face > 60:
    scene = "Multi-speaker discussion"
    recommended = "Transformer or FCAI adaptive"
elif avg_faces < 1.5 and pct_multi_face < 30:
    scene = "Interview/Single speaker"
    recommended = "LR-ASD baseline (high accuracy)"
else:
    scene = "Mixed/Unknown"
    recommended = "FCAI adaptive (safe choice)"

print(f"  Scene type:      {scene}")
print(f"  Avg faces:       {avg_faces:.2f}")
print(f"  Multi-face %:    {pct_multi_face:.1f}%")
print(f"  Recommended:     {recommended}")

# ── Simulate routing on Columbia ─────────────────────────────────────────────
# Load scores from different models
score_paths = {
    'LR-ASD':      f'{COLDATA}/col/pywork_baseline/scores.pckl',
    'Transformer': f'{COLDATA}/col/pywork/scores_ablation1.pckl',
    'FCAI':        f'{COLDATA}/col/pywork_fcai/scores.pckl',
}

# Compute what each model gets per frame based on face count
# Route: 1 face → baseline, 2+ faces → transformer
print("\n=== FRAME-LEVEL ROUTING ANALYSIS ===")
single_face_frames = sum(1 for c in all_face_counts if c == 1)
multi_face_frames  = sum(1 for c in all_face_counts if c >= 2)
total = len(all_face_counts)
print(f"Single-face frames: {single_face_frames} ({100*single_face_frames/total:.1f}%)")
print(f"Multi-face frames:  {multi_face_frames}  ({100*multi_face_frames/total:.1f}%)")
print(f"→ FCAI uses Transformer for {100*multi_face_frames/total:.1f}% of frames")
print(f"→ FCAI uses LR-ASD for {100*single_face_frames/total:.1f}% of frames")

# ── Visualization ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Face count distribution
ax = axes[0]
counts = [sum(1 for c in all_face_counts if c==n) for n in [1,2,3,4]]
ax.bar(['1','2','3','4+'], counts, color=['#2196F3','#FF9800','#F44336','#9C27B0'])
ax.set_xlabel('Faces per Frame')
ax.set_ylabel('Number of Frames')
ax.set_title('Columbia: Face Count Distribution')
ax.grid(True, alpha=0.3, axis='y')
for i, (x, c) in enumerate(zip([1,2,3,'4+'], counts)):
    ax.text(i, c+50, f'{c}', ha='center', fontsize=10)

# Plot 2: Columbia F1 comparison
ax = axes[1]
model_names = ['TalkNet', 'LR-ASD', 'Transformer', 'Multi-face', 'FCAI\n(adaptive)']
f1_scores   = [62.59, 66.24, 71.25, 64.37, 70.96]
colors_bar  = ['#9E9E9E','#2196F3','#F44336','#FF9800','#4CAF50']
bars = ax.bar(model_names, f1_scores, color=colors_bar)
ax.set_ylabel('Columbia F1 (%)')
ax.set_title('Columbia F1: Key Models')
ax.set_ylim(55, 78)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

# Plot 3: Routing pie chart
ax = axes[2]
ax.pie([single_face_frames, multi_face_frames],
       labels=[f'Single face\n→ LR-ASD\n({100*single_face_frames/total:.0f}%)',
               f'Multi-face\n→ Transformer\n({100*multi_face_frames/total:.0f}%)'],
       colors=['#2196F3','#F44336'],
       autopct='%1.1f%%', startangle=90, textprops={'fontsize':11})
ax.set_title('FCAI Frame Routing\n(Columbia)')

plt.suptitle('Scene-Type Routing: Face Statistics Drive Model Selection',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('scene_type_routing.png', dpi=150, bbox_inches='tight')
print("\nSaved scene_type_routing.png")
