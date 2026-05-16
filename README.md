Here is the README text:

---

# ASD Generalization Study

**Course:** CS671 Deep Learning | **Group:** 10 | **Guide:** Dr. Jyoti Nigam | **Prof:** Aditya Nigam

---

## Problem Statement

Active Speaker Detection (ASD) is the task of determining which person is speaking at each moment in a video containing multiple visible faces. The system receives synchronized audio and video and produces a per-frame binary speaking label for every visible face.

Our research question: **Why do lightweight single-candidate ASD models generalize better across datasets than heavier multi-candidate models?** We identify the exact architectural mechanism, prove it empirically through controlled ablations, attempt to fix it through six principled approaches, and build a practical system that exploits the finding.

---

## Key Finding

Multi-face inter-speaker context is the dominant driver of domain overfitting in ASD. Models that see other faces during training learn spurious co-occurrence patterns from Hollywood movie editing rhythms that do not transfer to naturalistic settings. Transformer temporal modeling reduces domain drop from **−27.87pp to −0.69pp**. Our FCAI adaptive system achieves **94.11% AVA mAP + 70.96% Columbia F1** simultaneously — the best combined result of any system we tested.

---

## Datasets

**AVA-ActiveSpeaker** — 120 Hollywood movies for training, 33 for validation. Per-frame binary speaking labels at 25 FPS. Face crops 112×112 pixels. Audio as 13-dim MFCC features. Metric: mAP. Download: https://research.google.com/ava/download.html

**Columbia** — 35-minute academic panel discussion, 5 speakers (Bell, Boll, Lieb, Long, Sick). Metric: F1 score. We use a frame-overlap evaluation protocol (FPS conversion 29.97→25fps) applied uniformly across all models due to video version coordinate mismatches with IOU-based GT boxes. Relative comparisons between all models are valid.

---

## Results

| Model | Params | AVA mAP | Columbia F1 | Domain Drop |
|-------|--------|---------|-------------|-------------|
| TalkNet | 15.01M | 92.15% | 62.59% | −29.56pp |
| LR-ASD baseline | 0.84M | 94.11% | 66.24% | −27.87pp |
| + Transformer temporal | 0.86M | 70.27% | **71.25%** | **−0.69pp** |
| + Multi-face mean | 0.84M | 82.87% | 64.37% | −18.50pp |
| + Attention context | 0.85M | 84.05% | 64.33% | −19.72pp |
| + Large capacity | 3.37M | 93.52% | 67.11% | −26.41pp |
| + Augmentation | 0.84M | 92.50% | 65.73% | −26.77pp |
| + Hard negatives | 0.84M | 91.88% | 64.93% | −26.95pp |
| TalkNCE | 0.88M | 68.25% | 67.11% | −1.14pp |
| CIR-020 | 0.84M | 85.94% | 65.88% | −20.06pp |
| **FCAI adaptive** | 1.70M | **94.11%** | **70.96%** | −23.15pp |
| IRM | 0.84M | 90.88% | 63.73% | −27.15pp |
| CaMIB | 0.87M | 89.58% | 64.49% | −25.09pp |
| SCSN | 0.84M | 94.11% | 67.11% | −27.00pp |

---

## Novel Contributions

**CIR — Correlation Independence Regularization**
Penalizes the absolute Pearson correlation between simultaneous face prediction scores during training. Designed to prevent the model from learning spurious inter-face co-occurrence patterns. Tested at λ=0.05, 0.10, 0.20. Files: `ASD_cir.py`, `train_cir_005.py`, `train_cir_010.py`, `train_cir_020.py`, `model_cir/`

**FCAI — Face-Count Adaptive Inference**
Inference-time routing system requiring no retraining. Routes single-face frames to LR-ASD baseline (best in-domain accuracy) and multi-face frames to Transformer (best generalization). Best practical system overall. Files: `ASD_adaptive.py`, `eval_fcai_ava.py`, `eval_fcai_columbia.py`

**TalkNCE — Contrastive Audio-Visual Loss**
Supervised contrastive loss applied to the transformer model. Pushes speaking frame embeddings apart from non-speaking embeddings across different clips, preventing shortcut learning. Files: `ASD_talknce.py`, `train_talknce.py`, `loss_talknce.py`, `model_talknce/`

**SCSN — Scene-Conditioned Score Calibration Network**
Uses CLIP ViT-B/32 to encode video frames into 512-dimensional scene embeddings. A 5000-parameter MLP learns to calibrate ASD scores conditioned on scene context. First use of vision-language embeddings for ASD score calibration. Files: `scsn_train.py`, `clip_embed_val.py`, `clip_scene_embed.py`, `eval_columbia_scsn.py`

**IRM — Invariant Risk Minimization (Negative Result)**
First application of IRM to ASD. Single-face frames as environment 1, multi-face frames as environment 2. Failed because AVA has only 13 single-face clips out of 1786 total — environments too imbalanced for IRM to work. Files: `ASD_irm.py`, `train_irm.py`, `eval_irm_tmp.py`, `run_scoring_irm.py`

**CaMIB — Causal Multimodal Information Bottleneck (Negative Result)**
Splits 128-dim embedding into causal and shortcut subspaces, predicting only from causal component. Failed because the bottleneck collapsed the score range from −4.9 to +2.7 down to −0.4 to +0.4, losing discriminative signal. Files: `ASD_camib.py`, `train_camib.py`, `eval_camib_tmp.py`, `run_scoring_camib.py`

**SCER — Spurious Correlation Embedding Regularization (Negative Result)**
Penalizes embedding distance between speaking frames in single vs multi-face environments. Loss was always zero because AVA has no single-face clips — the environments required to compute the loss do not exist in AVA. Files: `ASD_scer.py`, `train_scer.py`

---

## Mechanistic Analysis

**Co-occurrence correlation** — Pearson correlation between simultaneous face prediction scores. Multi-face models: r=+0.59 (strong spurious dependency). Transformer: r=−0.09 (near-zero, no inter-face learning). File: `cooccurrence_analysis.py`

**Face-count breakdown** — Multi-face model mAP by number of visible faces: 84.77% (1 face) → 64.41% (2 faces) → 43.08% (3+ faces). LR-ASD baseline: 96.89% → 93.53% → 84.30%. File: `face_count_ava_breakdown.py`

**GradCAM visualization** — Transformer attends specifically to lip/mouth region. Multi-face model shows diffuse scattered activation. File: `gradcam_viz.py`

**Calibration ECE** — Expected Calibration Error on Columbia. TalkNCE: 0.097 (best). Multi-face: 0.237 (worst). File: `uncertainty_ece.py`

**Scene-type characterization** — Automatic domain characterizer using face statistics. Columbia: 2.11 avg faces/frame, 79.6% multi-face frames, 15.9% speaking overlap → correctly classified as multi-speaker discussion. File: `scene_type_routing.py`

All result figures are in `results/`.

---

## Repository Structure

```
ASD_*.py                    model wrappers for each variant
train_*.py                  training scripts
dataLoader_*.py             data loaders
eval_columbia_*.py          Columbia F1 evaluation scripts
run_scoring*.py             Columbia inference pipeline
model_*/                    model architectures (Encoder, Classifier, Model)
utils/                      AVA evaluation utilities
cooccurrence_analysis.py    inter-face correlation measurement
gradcam_viz.py              GradCAM activation visualization
face_count_ava_breakdown.py mAP by number of visible faces
scatter_correlation_drop.py correlation vs domain drop scatter plot
scene_type_routing.py       automatic scene-type characterizer
uncertainty_ece.py          Expected Calibration Error analysis
speaker_heatmap.py          per-speaker domain drop heatmap
scsn_train.py               SCSN training script
clip_embed_val.py           CLIP embedding for AVA val clips
clip_scene_embed.py         CLIP scene embedding for Columbia
loss.py                     AV and visual loss functions
loss_talknce.py             TalkNCE contrastive loss
results/                    figures, plots, analysis outputs
```

---

## Setup

```bash
conda create -n asd_exp python=3.8
conda activate asd_exp
pip install torch torchaudio opencv-python python_speech_features \
            pandas tqdm scikit-learn scipy matplotlib seaborn \
            clip transformers
```

---

## Training

```bash
# Baseline LR-ASD
python train.py --dataPathAVA /path/to/AVADataPath --savePath exps/baseline

# Transformer ablation
python train_ablation1_transformer.py --dataPathAVA /path/to/AVADataPath --savePath exps/transformer

# FCAI (no training needed — uses two existing models)
python eval_fcai_ava.py --dataPathAVA /path/to/AVADataPath

# CIR variants
python train_cir_020.py --dataPathAVA /path/to/AVADataPath --savePath exps/cir_020 --lambda_cir 0.20
```

---

## Evaluation

```bash
# Columbia F1 (after running run_scoring.py to generate scores.pckl)
python run_scoring.py --pretrainModel exps/baseline/model/best.model \
    --pycropPath /path/to/ColData/col/pycrop \
    --pyworkPath /path/to/ColData/col/pywork

python eval_columbia_final.py
```

---

## Acknowledgements

Built on top of [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) (Liao et al., IJCV 2025) and [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) (Tao et al., ACM MM 2021).