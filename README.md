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

Prepare Columbia in the TalkNet/LR-ASD layout under `ColData/col/` (`pyframes/`, `pycrop/`, `pywork/`, `col_labels/`). Full end-to-end preprocessing is in `Columbia_test.py` (face detection, tracking, cropping).

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

### Script map (reproduce each row)

| Model | Train | ASD wrapper | Model code | Columbia scoring | Columbia eval |
|-------|-------|-------------|------------|------------------|---------------|
| LR-ASD baseline | `train.py` | `ASD.py` | `model/` | `run_scoring.py` → `ColData/col/pywork/` | `eval_columbia_final.py` |
| + Transformer | `train_ablation1_transformer.py` | `ASD_transformer.py` | `model_transformer/` | `run_scoring.py` → `pywork/` + `scores_ablation1.pckl` | `eval_columbia_final.py` |
| + Multi-face mean | `train_ablation2.py` | `ASD_ablation2.py` | `model_ablation2/` | `run_scoring.py` → `pywork_ablation3/` | `eval_columbia_final.py` |
| + Attention context | `train_attn_context.py` | `ASD_attn_context.py` | `model_attn_context/` | `run_scoring_attn.py` | `eval_columbia_final.py` |
| + Large capacity | `train_ablation3_large.py` | `ASD_large.py` | `model_large/` | `run_scoring.py` → `pywork_ablation3/` | `eval_columbia_final.py` |
| + Augmentation | `train_improved.py` | `ASD.py` | `model/` | `run_scoring.py` → `pywork_improved/` | `eval_columbia_final.py` |
| + Hard negatives | `train_hardneg.py` | `ASD.py` | `model/` | `run_scoring.py` → `pywork_hardneg/` | `eval_columbia_final.py` |
| TalkNCE | `train_talknce.py` | `ASD_talknce.py` | `model_talknce/` | `run_scoring.py` → `pywork_talknce/` | `eval_columbia_final.py` |
| CIR-020 | `train_cir_020.py` | `ASD_cir.py` | `model_cir/` | `run_scoring.py` → `pywork_cir020/` | `eval_columbia_final.py` |
| FCAI adaptive | — (inference only) | `ASD_adaptive.py` | baseline + transformer | `eval_fcai_columbia.py` | `eval_columbia_final.py` (`--pyworkPath` → `pywork_fcai`) |
| IRM | `train_irm.py` | `ASD_irm.py` | `model_ablation2/` * | `run_scoring_irm.py` | `eval_columbia_irm.py` |
| CaMIB | `train_camib.py` | `ASD_camib.py` | `model_ablation2/` * | `run_scoring_camib.py` | `eval_columbia_camib.py` |
| SCSN | `scsn_train.py` | — | small MLP on CLIP | `eval_columbia_scsn.py` | (calibrated scores) |
| SCER | `train_scer.py` | `ASD_scer.py` | `model_ablation2/` * | same as multi-face | `eval_columbia_final.py` |
| TalkNet | external | TalkNet-ASD | — | see `TalkNet-exps/` | `eval_columbia_final.py` |

\* IRM, CaMIB, and SCER reuse the multi-face backbone in `model_ablation2/` with different training objectives in `ASD_irm.py`, `ASD_camib.py`, `ASD_scer.py`.

---

## Novel Contributions

**CIR — Correlation Independence Regularization**
Penalizes the absolute Pearson correlation between simultaneous face prediction scores during training. Files: `ASD_cir.py`, `train_cir_005.py`, `train_cir_010.py`, `train_cir_020.py`, `model_cir/`

**FCAI — Face-Count Adaptive Inference**
Inference-time routing: single-face → LR-ASD baseline, multi-face → Transformer. Files: `ASD_adaptive.py`, `eval_fcai_ava.py`, `eval_fcai_columbia.py`

**TalkNCE — Contrastive Audio-Visual Loss**
Files: `ASD_talknce.py`, `train_talknce.py`, `loss_talknce.py`, `model_talknce/`

**SCSN — Scene-Conditioned Score Calibration Network**
CLIP ViT-B/32 scene embeddings + small MLP. Files: `scsn_train.py`, `clip_embed_val.py`, `clip_scene_embed.py`, `eval_columbia_scsn.py`

**IRM / CaMIB / SCER (negative results)** — see script map above; Columbia eval: `eval_columbia_irm.py`, `eval_columbia_camib.py`.

---

## Mechanistic Analysis

| Analysis | Script | Output |
|----------|--------|--------|
| Co-occurrence correlation | `cooccurrence_analysis.py` | `results/cooccurrence_results.csv` |
| Face-count mAP breakdown | `face_count_ava_breakdown.py` | `results/face_count_ava_breakdown.png` |
| GradCAM | `gradcam_viz.py` | `results/gradcam_comparison.png` |
| Calibration ECE | `uncertainty_ece.py` | `results/uncertainty_ece.png` |
| Scene-type routing | `scene_type_routing.py` | `results/scene_type_routing.png` |
| Correlation vs domain drop | `scatter_correlation_drop.py` | `results/scatter_correlation_drop.png` |
| Per-speaker F1 heatmap | `speaker_heatmap.py` | `results/speaker_heatmap.png` |

`cooccurrence_analysis.py` skips missing `scores.pckl` files; TalkNet requires `ColData/col/pywork_talknet/scores.pckl` if included.

All figures and CSVs live in **`results/`** only.

---

## Repository Structure

```
ASD_*.py                    model wrappers (one per variant)
train_*.py                  AVA training
dataLoader_*.py             loaders for ablations
eval_columbia_*.py          Columbia F1 (see eval script roles below)
run_scoring*.py             write scores.pckl from a checkpoint
Columbia_test.py            full Columbia demo pipeline (detect → track → score)
model_*/                    encoders / classifiers (shared S3FD under model/faceDetector/)
utils/                      AVA metrics + repo_paths.py (default data locations)
results/                    figures, CSVs, eval summary append log
TalkNet-exps/               TalkNet baseline logs (see TalkNet-exps/README.md)
requirements.txt            Python dependencies
LICENSE                     MIT + upstream attribution
```

### Columbia evaluation scripts

| Script | Role |
|--------|------|
| `eval_columbia_final.py` | **Canonical** fps-corrected frame-overlap F1 (used in paper table) |
| `eval_columbia_iou.py` | IOU-based track–GT matching (legacy; coordinate mismatch on our video) |
| `eval_columbia_quick.py` | Fast rescoring from existing `tracks.pckl` |
| `eval_columbia_smoothed.py` | Same protocol with temporal score smoothing |
| `eval_columbia_scsn.py` | F1 with SCSN-calibrated scores |
| `eval_columbia_camib.py` / `eval_columbia_irm.py` | F1 for CaMIB / IRM `pywork_*` dirs |

---

## Artifacts not in git

| Item | Location / action |
|------|-------------------|
| AVA preprocessed data | `--dataPathAVA` (see LR-ASD README) |
| Columbia crops & labels | `ColData/` at repo root, or set `COL_DATA_ROOT` |
| S3FD face detector weights | `model/faceDetector/s3fd/sfd_face.pth` (~95 MB; from [face-detection-pytorch](https://github.com/cs-giung/face-detection-pytorch) / TalkNet-ASD) |
| LR-ASD AVA pretrained (optional) | `weight/pretrain_AVA.model` |
| Trained checkpoints | `exps/<experiment>/model/*.model` (gitignored) |
| Columbia `scores.pckl` | produced by `run_scoring*.py` |
| CLIP embeddings | `clip_embeddings/` (from `clip_embed_val.py`, `clip_scene_embed.py`) |
| Large TalkNet CSVs | not stored; see `TalkNet-exps/README.md` |

Default paths resolve via `utils/repo_paths.py` (repo root + `ColData/`). Override with environment variables `COL_DATA_ROOT` and `CLIP_EMBED_DIR`.

---

## Setup

```bash
conda create -n asd_exp python=3.8
conda activate asd_exp
pip install -r requirements.txt
```

Place `sfd_face.pth` under `model/faceDetector/s3fd/` before running `Columbia_test.py` or any face-detection step.

---

## Training

```bash
# Baseline LR-ASD
python train.py --dataPathAVA /path/to/AVADataPath --savePath exps/baseline

# Transformer ablation
python train_ablation1_transformer.py --dataPathAVA /path/to/AVADataPath --savePath exps/transformer

# Multi-face mean pooling
python train_ablation2.py --dataPathAVA /path/to/AVADataPath --savePath exps/ablation2_multiface

# CIR
python train_cir_020.py --dataPathAVA /path/to/AVADataPath --savePath exps/cir_020 --lambda_cir 0.20
```

---

## Evaluation

```bash
# 1) Generate Columbia scores (example: baseline)
python run_scoring.py --pretrainModel exps/baseline/model/best.model \
    --pycropPath ColData/col/pycrop \
    --pyworkPath ColData/col/pywork

# 2) Canonical Columbia F1
python eval_columbia_final.py \
    --colSavePath ColData \
    --pyworkPath ColData/col/pywork \
    --pyframesPath ColData/col/pyframes

# FCAI: build scores then evaluate pywork_fcai
python eval_fcai_columbia.py
python eval_columbia_final.py --pyworkPath ColData/col/pywork_fcai
```

---

## TalkNet baseline

Comparison numbers and training logs are summarized under `TalkNet-exps/` (not full per-frame exports). See `TalkNet-exps/README.md`.

---

## Acknowledgements

Built on [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) (Liao et al., IJCV 2025) and [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) (Tao et al., ACM MM 2021).

## License

See [LICENSE](LICENSE).
