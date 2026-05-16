import os, glob, pickle, torch, clip, cv2, numpy as np
from tqdm import tqdm

DEVICE = "cuda:0"  # 43GB free
AVA_VIDEO_ROOT = "/usershome/cs671_user6/asd_project/LR-ASD/data/clips_videos/train"
COL_FRAMES_PATH = "/usershome/cs671_user6/asd_project/ColData/col/pyframes"
SAVE_DIR = "/usershome/cs671_user6/asd_project/LR-ASD/clip_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()
print("CLIP loaded on", DEVICE)

def embed_frames(frame_paths, n_sample=5):
    if len(frame_paths) == 0:
        return None
    idxs = np.linspace(0, len(frame_paths)-1, min(n_sample, len(frame_paths)), dtype=int)
    imgs = []
    for i in idxs:
        img = cv2.imread(frame_paths[i])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(img)
        imgs.append(preprocess(pil))
    if len(imgs) == 0:
        return None
    batch = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.mean(dim=0).cpu().numpy().astype(np.float32)

# --- Columbia embedding (single video) ---
print("\nEmbedding Columbia...")
col_frames = sorted(glob.glob(os.path.join(COL_FRAMES_PATH, "*.jpg")))
col_emb = embed_frames(col_frames, n_sample=10)
np.save(os.path.join(SAVE_DIR, "columbia_clip.npy"), col_emb)
print(f"Columbia embedding shape: {col_emb.shape}")

# --- AVA train clips embedding ---
print("\nEmbedding AVA train clips...")
video_dirs = sorted(glob.glob(os.path.join(AVA_VIDEO_ROOT, "*")))
ava_embeddings = {}
for vdir in tqdm(video_dirs):
    vid_id = os.path.basename(vdir)
    # each subdir has entity clips as subdirs
    entity_dirs = sorted(glob.glob(os.path.join(vdir, "*")))
    for edir in entity_dirs:
        frames = sorted(glob.glob(os.path.join(edir, "*.jpg")))
        if len(frames) < 5:
            continue
        key = f"{vid_id}/{os.path.basename(edir)}"
        emb = embed_frames(frames, n_sample=5)
        if emb is not None:
            ava_embeddings[key] = emb

print(f"Total AVA clips embedded: {len(ava_embeddings)}")
with open(os.path.join(SAVE_DIR, "ava_train_clip_embeddings.pkl"), "wb") as f:
    pickle.dump(ava_embeddings, f)
print("Saved ava_train_clip_embeddings.pkl")
