import os, sys, cv2, torch, numpy as np, glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/usershome/cs671_user6/asd_project/LR-ASD')

# ── helpers ──────────────────────────────────────────────────────────────────
def load_face_frames(pycrop_path, track_idx, n=16):
    """Load first n grayscale frames from a face track .avi"""
    files = sorted(glob.glob(f'{pycrop_path}/*.avi'))
    if track_idx >= len(files):
        return None, None
    path = files[track_idx]
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ret, f = cap.read()
        if not ret: break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (112, 112))
        frames.append(gray)
    cap.release()
    if len(frames) == 0:
        return None, None
    arr = np.array(frames, dtype=np.float32)   # (T, 112, 112)
    return arr, path

def get_gradcam(model, visual_input, target_layer):
    """
    Compute GradCAM for visual encoder.
    visual_input: (1, T, 112, 112) float tensor on cuda
    Returns heatmap (112, 112)
    """
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    model.eval()
    visual_input.requires_grad_(True)

    # Forward
    B, T, W, H = visual_input.shape
    x = visual_input.view(B, 1, T, W, H)
    x = (x / 255 - 0.4161) / 0.1688
    out = model.visualEncoder(x)  # (B, T, C)

    # Use mean of output as scalar for backward
    score = out.mean()
    model.zero_grad()
    score.backward()

    fh.remove()
    bh.remove()

    if len(gradients) == 0 or len(activations) == 0:
        return np.zeros((112, 112))

    grad = gradients[0]        # (B, C, T, H, W) or similar
    act  = activations[0]

    # Pool over channels and time
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
    cam = (weights * act).sum(dim=1).squeeze()         # (T, H, W)
    cam = torch.relu(cam).mean(dim=0)                  # (H, W)
    cam = cam.detach().cpu().numpy()

    # Resize to 112x112
    cam = cv2.resize(cam, (112, 112))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_heatmap(gray_frame, heatmap):
    """Overlay heatmap on grayscale face frame"""
    rgb = cv2.cvtColor(gray_frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    heat_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.5, heat_colored, 0.5, 0)
    return overlay

# ── load models ──────────────────────────────────────────────────────────────
print("Loading models...")

from ASD import ASD as ASD_baseline
from ASD_transformer import ASD as ASD_trans

baseline = ASD_baseline()
baseline.loadParameters('exps/exp_baseline/model/model_0022.model')
baseline.eval()

transformer = ASD_trans()
transformer.loadParameters('exps/ablation1_transformer/model/model_0039.model')
transformer.eval()

# Multi-face model
from ASD_ablation2 import ASD as ASD_multi
multi = ASD_multi()
multi.loadParameters('exps/ablation2_multiface/model/model_0008.model')
multi.eval()

print("All models loaded")

# ── find good frames ──────────────────────────────────────────────────────────
PYCROP = '/usershome/cs671_user6/asd_project/ColData/col/pycrop'
import pickle
tracks  = pickle.load(open('/usershome/cs671_user6/asd_project/ColData/col/pywork/tracks.pckl','rb'))
scores_baseline = pickle.load(open('/usershome/cs671_user6/asd_project/ColData/col/pywork_baseline/scores.pckl','rb'))

# Find track with high speaking score (clear speaker)
best_speaking_idx = None
best_speaking_score = -999
for i, sc in enumerate(scores_baseline):
    if len(sc) > 0 and max(sc) > best_speaking_score:
        best_speaking_score = max(sc)
        best_speaking_idx = i

print(f"Best speaking track: {best_speaking_idx}, max score: {best_speaking_score:.2f}")

# Use tracks 0-5 for visualization variety
viz_tracks = [best_speaking_idx, best_speaking_idx+1, best_speaking_idx+2]
viz_tracks = [t for t in viz_tracks if t < len(tracks)][:3]
print(f"Visualizing tracks: {viz_tracks}")

# ── target layers ────────────────────────────────────────────────────────────
target_baseline     = baseline.model.visualEncoder.block3.t_2
target_transformer  = transformer.model.visualEncoder.block3.t_2
target_multi        = multi.model.visualEncoder.block3.t_2

# ── generate visualizations ──────────────────────────────────────────────────
n_tracks = len(viz_tracks)
fig, axes = plt.subplots(n_tracks, 4, figsize=(16, 4*n_tracks))
if n_tracks == 1:
    axes = axes[np.newaxis, :]

col_labels = ['Original Face', 'LR-ASD Baseline', 'Transformer', 'Multi-face']

for row, track_idx in enumerate(viz_tracks):
    frames, path = load_face_frames(PYCROP, track_idx, n=16)
    if frames is None:
        continue

    # Use middle frame for visualization
    mid = len(frames) // 2
    face_frame = frames[mid]  # (112, 112)

    # Prepare tensor
    vf = torch.FloatTensor(frames).unsqueeze(0).cuda()  # (1, T, 112, 112)

    # GradCAM for each model
    cam_baseline    = get_gradcam(baseline.model,    vf, target_baseline)
    cam_transformer = get_gradcam(transformer.model, vf, target_transformer)
    cam_multi       = get_gradcam(multi.model,       vf, target_multi)

    # Plot
    axes[row, 0].imshow(face_frame, cmap='gray')
    axes[row, 0].set_title(f'Track {track_idx}\nOriginal', fontsize=10)
    axes[row, 0].axis('off')

    for col, (cam, label) in enumerate([
        (cam_baseline, 'LR-ASD Baseline'),
        (cam_transformer, 'Transformer'),
        (cam_multi, 'Multi-face'),
    ], start=1):
        overlay = overlay_heatmap(face_frame, cam)
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(label, fontsize=10)
        axes[row, col].axis('off')

    # Add score annotation
    if track_idx < len(scores_baseline) and len(scores_baseline[track_idx]) > mid:
        score = scores_baseline[track_idx][mid]
        axes[row, 0].set_xlabel(f'Score: {score:.2f}', fontsize=9)

plt.suptitle('GradCAM: What Each Model Attends To\n(Red = high attention, Blue = low attention)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

out_path = '/usershome/cs671_user6/asd_project/LR-ASD/gradcam_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
