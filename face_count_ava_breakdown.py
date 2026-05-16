import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import os

models = {
    'LR-ASD':      'exps/exp_baseline/val_res.csv',
    'Transformer': 'exps/ablation1_transformer/val_res.csv',
    'Multi-face':  'exps/ablation2_multiface/val_res.csv',
    'Large':       'exps/ablation3_large/val_res.csv',
    'Augment':     'exps/improved/val_res.csv',
    'Hard-neg':    'exps/hardneg/val_res.csv',
    'Attn-ctx':    'exps/attn_context/val_res.csv',
}

# Load GT with real labels
gt_df = pd.read_csv('data/csv/val_orig.csv')
# label_id: 0=not speaking, 1=speaking, 2=unknown — drop unknowns
gt_df = gt_df[gt_df['label_id'] != 2][['video_id','frame_timestamp','entity_id','label_id']]

# Count faces per (video_id, frame_timestamp) from GT
frame_face_counts = gt_df.groupby(['video_id','frame_timestamp'])['entity_id'].count().reset_index()
frame_face_counts.columns = ['video_id','frame_timestamp','n_faces']

def compute_map_by_face_count(csv_path):
    pred_df = pd.read_csv(csv_path)[['video_id','frame_timestamp','entity_id','score']]
    
    # Merge with GT labels
    merged = pred_df.merge(gt_df, on=['video_id','frame_timestamp','entity_id'], how='inner')
    # Merge with face counts
    merged = merged.merge(frame_face_counts, on=['video_id','frame_timestamp'], how='left')
    
    results = {}
    for n, label in [(1,'1 face'), (2,'2 faces'), (3,'3+ faces')]:
        if n < 3:
            subset = merged[merged['n_faces'] == n]
        else:
            subset = merged[merged['n_faces'] >= 3]
        
        if len(subset) < 100:
            results[label] = None
            continue
        
        labels = subset['label_id'].values
        scores = subset['score'].values
        ap = average_precision_score(labels, scores) * 100
        results[label] = ap
        print(f"  {label}: {len(subset):7d} samples, mAP={ap:.2f}%")
    
    return results

print("Computing mAP by face count...")
all_results = {}
for mname, path in models.items():
    if not os.path.exists(path):
        print(f"{mname}: MISSING")
        continue
    print(f"\n{mname}:")
    all_results[mname] = compute_map_by_face_count(path)

# Plot
face_labels = ['1 face', '2 faces', '3+ faces']
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(face_labels))
width = 0.12
colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

for i, (mname, results) in enumerate(all_results.items()):
    vals = [results.get(fl, 0) or 0 for fl in face_labels]
    bars = ax.bar(x + i*width - (len(all_results)*width/2), vals,
                  width, label=mname, color=colors[i])

ax.set_xlabel('Number of Visible Faces per Frame', fontsize=12)
ax.set_ylabel('AVA Val mAP (%)', fontsize=12)
ax.set_title('AVA Val mAP by Face Count\nDoes multi-face context help more when more faces are visible?',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(face_labels, fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(60, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
import os
from utils.repo_paths import results_dir

out_path = os.path.join(results_dir(), "face_count_ava_breakdown.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
