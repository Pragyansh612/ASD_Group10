import os, pickle, torch, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

DEVICE = "cuda:0"
from utils.repo_paths import clip_embeddings_dir, exps_path

EMBED_DIR = clip_embeddings_dir()
VAL_RES   = os.path.join(exps_path("exp_baseline"), "val_res.csv")
SAVE_PATH = os.path.join(EMBED_DIR, "scsn.pt")
T = 50

print("Loading AVA val CLIP embeddings...")
with open(os.path.join(EMBED_DIR, "ava_val_clip_embeddings.pkl"), "rb") as f:
    ava_embs = pickle.load(f)
print(f"Embeddings: {len(ava_embs)}")

print("Loading val_res.csv...")
df = pd.read_csv(VAL_RES)
grouped = df.sort_values('frame_timestamp').groupby('entity_id')

class SCSNDataset(Dataset):
    def __init__(self, emb_by_clipid, grouped, T=50):
        self.samples = []
        matched = 0
        for entity_id, group in grouped:
            if entity_id not in emb_by_clipid:
                continue
            emb    = emb_by_clipid[entity_id]
            scores = group['score'].values.astype(np.float32)
            labels = (scores > 0).astype(np.float32)
            if len(scores) <= T:
                continue
            n_windows = max(1, len(scores) // T)
            for _ in range(n_windows):
                start = np.random.randint(0, len(scores) - T)
                self.samples.append((emb,
                                     scores[start:start+T],
                                     labels[start:start+T]))
            matched += 1
        print(f"Matched clips: {matched}, total windows: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        emb, scores, labels = self.samples[idx]
        return torch.FloatTensor(emb), torch.FloatTensor(scores), torch.FloatTensor(labels)

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

print("Building dataset...")
full_dataset = SCSNDataset(ava_embs, grouped, T=T)
if len(full_dataset) == 0:
    print("ERROR: 0 samples"); exit(1)

# 80/20 train/val split
n_val   = max(1, int(0.2 * len(full_dataset)))
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
print(f"Train: {n_train}  Val: {n_val}")

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4)

model  = SCSN(clip_dim=512, T=T, dropout=0.3).to(DEVICE)
optim  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5)
crit   = nn.BCEWithLogitsLoss()

best_val_loss = float('inf')
best_epoch    = 0
patience      = 5

print("Training SCSN...")
for epoch in range(1, 15):
    # train
    model.train()
    tr_loss = 0
    for emb, scores, labels in train_loader:
        emb, scores, labels = emb.to(DEVICE), scores.to(DEVICE), labels.to(DEVICE)
        loss = crit(model(emb, scores), labels)
        optim.zero_grad(); loss.backward(); optim.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    # val
    model.eval()
    vl_loss = 0
    with torch.no_grad():
        for emb, scores, labels in val_loader:
            emb, scores, labels = emb.to(DEVICE), scores.to(DEVICE), labels.to(DEVICE)
            vl_loss += crit(model(emb, scores), labels).item()
    vl_loss /= len(val_loader)
    sched.step(vl_loss)

    print(f"Epoch {epoch:3d}  TrainLoss: {tr_loss:.4f}  ValLoss: {vl_loss:.4f}  LR: {optim.param_groups[0]['lr']:.6f}")

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_epoch    = epoch
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  --> Best model saved (val_loss={best_val_loss:.4f})")

    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch} (best={best_epoch})")
        break

print(f"\nBest epoch: {best_epoch}  ValLoss: {best_val_loss:.4f}")
print(f"SCSN saved to {SAVE_PATH}")
