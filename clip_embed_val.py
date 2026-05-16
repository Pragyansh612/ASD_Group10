import os, glob, pickle, torch, clip, cv2, numpy as np
from tqdm import tqdm

DEVICE = "cuda:0"
import argparse
from utils.repo_paths import REPO_ROOT, clip_embeddings_dir

parser = argparse.ArgumentParser()
parser.add_argument("--avaVideoRoot", type=str, default=os.path.join(REPO_ROOT, "data", "clips_videos", "val"))
parser.add_argument("--saveDir", type=str, default=None)
args = parser.parse_args()

AVA_VIDEO_ROOT = args.avaVideoRoot
SAVE_DIR = args.saveDir or clip_embeddings_dir()
os.makedirs(SAVE_DIR, exist_ok=True)

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def embed_frames(frame_paths, n_sample=5):
    from PIL import Image
    idxs = np.linspace(0, len(frame_paths)-1, min(n_sample, len(frame_paths)), dtype=int)
    imgs = []
    for i in idxs:
        img = cv2.imread(frame_paths[i])
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(preprocess(Image.fromarray(img)))
    if not imgs: return None
    batch = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.mean(dim=0).cpu().numpy().astype(np.float32)

print("Embedding AVA val clips...")
video_dirs = sorted(glob.glob(os.path.join(AVA_VIDEO_ROOT, "*")))
ava_val_embeddings = {}
for vdir in tqdm(video_dirs):
    vid_id = os.path.basename(vdir)
    for edir in sorted(glob.glob(os.path.join(vdir, "*"))):
        frames = sorted(glob.glob(os.path.join(edir, "*.jpg")))
        if len(frames) < 5: continue
        # key matches val_res.csv entity_id: VIDEOID_startSEC_endSEC:ENTITY
        key = os.path.basename(edir)  # e.g. HV0H6oc4Kvs_0960_1020:1
        emb = embed_frames(frames, n_sample=5)
        if emb is not None:
            ava_val_embeddings[key] = emb

print(f"Total val clips embedded: {len(ava_val_embeddings)}")
with open(os.path.join(SAVE_DIR, "ava_val_clip_embeddings.pkl"), "wb") as f:
    pickle.dump(ava_val_embeddings, f)
print("Saved ava_val_clip_embeddings.pkl")

# debug: print first 3 keys
print("Sample keys:", list(ava_val_embeddings.keys())[:3])
