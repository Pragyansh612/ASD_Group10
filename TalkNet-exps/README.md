# TalkNet baseline artifacts

TalkNet-ASD comparison runs (training logs and small summaries only).

| File | Description |
|------|-------------|
| `exp_baseline/score.txt` | Training log (epoch / mAP) |
| `exp_baseline/training_log.txt` | Same as `score.txt` (renamed copy for clarity) |
| `*.log` | Columbia scoring / eval logs (local only; not required for reproduction) |

Large per-frame CSV outputs (e.g. `val_res.csv`) are **not** stored in git. Regenerate with TalkNet-ASD on AVA if needed.

Reported TalkNet numbers in the main README: **92.15% AVA mAP**, **62.59% Columbia F1**.
