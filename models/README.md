# models/

Checkpoints are **not** stored in git (`*.pt` is ignored). Publish them as GitHub Release assets and download into this folder.

| File | What | Size |
|---|---|---|
| `hindi_finetuned_phase3.pt` | Hindi model after reasoning fine-tuning (used by `make eval-hindi`) | ≈ 97 MB |
| Base checkpoints (`hindi_model.pt`, `nepal_model.pt`) | Pretrained models, 25,345,216 parameters each | on the project's Google Drive; attach to a Release |

Load with `indic_lm.evaluation.reasoning.load_model`, which accepts both the current and the legacy key names.
