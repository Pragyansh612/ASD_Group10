"""
Columbia evaluation with SCSN-calibrated scores.
Same fps-based protocol as eval_columbia_final.py — only scores are calibrated.
"""
import os, glob, pickle, torch, numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn

DEVICE = "cuda:0"
EMBED_DIR = "/usershome/cs671_user6/asd_project/LR-ASD/clip_embeddings"
SCSN_PATH = os.path.join(EMBED_DIR, "scsn.pt")
pyworkPath   = '/usershome/cs671_user6/asd_project/ColData/col/pywork_baseline'
pyframesPath = '/usershome/cs671_user6/asd_project/ColData/col/pyframes'
colSavePath  = '/usershome/cs671_user6/asd_project/ColData'
T = 50

ORIG_FPS = 29.97; OUR_FPS = 25.0; RATIO = OUR_FPS / ORIG_FPS

# --- Load SCSN ---
class SCSN(nn.Module):
    def __init__(self, clip_dim=512, T=50, dropout=0.3):
        super().__init__()
        self.scene_enc = nn.Sequential(
            nn.Linear(clip_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),       nn.ReLU(), nn.Dropout(dropout)
        )
        self.score_enc = nn.Sequential(
            nn.Linear(T, 128),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.calibrate = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, T)
        )
    def forward(self, scene_emb, raw_scores):
        s = self.scene_enc(scene_emb)
        r = self.score_enc(raw_scores)
        return self.calibrate(torch.cat([s, r], dim=-1))

scsn = SCSN().to(DEVICE)
scsn.load_state_dict(torch.load(SCSN_PATH, map_location=DEVICE))
scsn.eval()
print("SCSN loaded")

# --- Load Columbia CLIP embedding ---
col_emb = np.load(os.path.join(EMBED_DIR, "columbia_clip.npy"))
col_emb_t = torch.FloatTensor(col_emb).unsqueeze(0).to(DEVICE)

# --- Load tracks and raw scores ---
vidTracks = pickle.load(open(os.path.join(pyworkPath, 'tracks.pckl'), 'rb'))
raw_scores = pickle.load(open(os.path.join(pyworkPath, 'scores.pckl'), 'rb'))

flist = sorted(glob.glob(os.path.join(pyframesPath, '*.jpg')))
first_frame = int(os.path.splitext(os.path.basename(flist[0]))[0])
total_frames = len(flist)

# --- Calibrate scores using SCSN ---
def calibrate_scores(score_seq, col_emb_t):
    """Apply SCSN calibration in sliding T-windows."""
    cal = np.array(score_seq, dtype=np.float32)
    L = len(score_seq)
    if L < T:
        window = np.zeros(T, dtype=np.float32)
        window[:L] = score_seq
        with torch.no_grad():
            out = scsn(col_emb_t, torch.FloatTensor(window).unsqueeze(0).to(DEVICE))
        cal = torch.sigmoid(out).squeeze().cpu().numpy()[:L] * 2 - 1
    else:
        calibrated = np.zeros(L, dtype=np.float32)
        counts = np.zeros(L, dtype=np.float32)
        for start in range(0, L - T + 1, T // 2):
            window = np.array(score_seq[start:start+T], dtype=np.float32)
            with torch.no_grad():
                out = scsn(col_emb_t, torch.FloatTensor(window).unsqueeze(0).to(DEVICE))
            cal_window = torch.sigmoid(out).squeeze().cpu().numpy() * 2 - 1
            calibrated[start:start+T] += cal_window
            counts[start:start+T] += 1
        counts = np.maximum(counts, 1)
        cal = calibrated / counts
    return cal.tolist()

print("Calibrating scores...")
calibrated_scores = []
for tidx, score in enumerate(raw_scores):
    cal = calibrate_scores(score, col_emb_t)
    calibrated_scores.append(cal)

# --- Rest is identical to eval_columbia_final.py ---
track_data = {}
for tidx, track in enumerate(vidTracks):
    score = calibrated_scores[tidx]
    fr, sc = [], []
    for fidx, frame in enumerate(track['track']['frame'].tolist()):
        if frame >= total_frames:
            continue
        s = float(np.mean(score[max(fidx-2,0): min(fidx+3,len(score)-1)]))
        actual_frame = frame + first_frame
        fr.append(actual_frame); sc.append(s)
    if len(fr) > 50:
        track_data[tidx] = (set(fr), fr, sc)

names = ['bell', 'boll', 'lieb', 'long', 'sick']
gt_per_speaker = {n: {} for n in names}
for name in names:
    with open(os.path.join(colSavePath, 'col_labels', 'fusion', f'{name}.txt')) as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            gt_frame_ours = int(int(parts[0]) * RATIO)
            gt = int(parts[4])
            if gt_frame_ours <= total_frames:
                gt_per_speaker[name][gt_frame_ours] = gt

gt_frame_sets = {n: set(gt_per_speaker[n].keys()) for n in names}
track_to_speaker = {}
all_assignments = []
for tidx, (frame_set, fr, sc) in track_data.items():
    for name in names:
        overlap = len(frame_set & gt_frame_sets[name])
        norm = overlap / max(len(gt_frame_sets[name]), 1)
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

print('\n=== COLUMBIA F1 RESULTS — SCSN CALIBRATED ===')
F1s = 0; valid = 0
for name in sorted(names):
    common = sorted(set(gt_per_speaker[name].keys()) & set(speaker_frame_scores[name].keys()))
    if len(common) < 100: continue
    preds  = np.array([1 if speaker_frame_scores[name][f] > -1 else 0 for f in common])
    labels = np.array([gt_per_speaker[name][f] for f in common])
    if labels.sum() == 0: continue
    F1  = f1_score(labels, preds, zero_division=0)
    ACC = accuracy_score(labels, preds)
    F1s += F1; valid += 1
    print(f'{name}: ACC={100*ACC:.2f}%  F1={100*F1:.2f}%')

avg_f1 = 100 * (F1s / valid) if valid > 0 else 0
print(f'\nSCSN Calibrated Columbia F1: {avg_f1:.2f}%')
print(f'Baseline (no calibration):   66.24%')
