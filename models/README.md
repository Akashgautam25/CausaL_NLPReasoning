# models/

This folder holds trained model checkpoints. They are **not stored in git** because the weights are large (about 270 MB, over a normal repository's comfortable size and close to GitHub's 100 MB per-file limit).

## distilbert_base_uncased/ (the project's trained model)

Fine-tuned `distilbert-base-uncased` for directional causal relation extraction
(labels: `NONE`, `CAUSE_E1_E2`, `CAUSE_E2_E1`).

* Trained on: SemEval-2010 Task 8 (Cause-Effect mapped to causal classes; all other relations mapped to NONE)
* Hardware: Apple M3 Pro (MPS), seed 42
* Settings: 4 epochs, batch size 16, learning rate 2e-5, max length 128
* Training time: 480 s; best validation macro-F1: 0.945 (epoch 3, the saved checkpoint)
* Gold test macro-F1: 0.942 (spaCy rules: 0.829, keyword heuristic: 0.508)

### Recreate it

    python -m dl.training.train --model distilbert-base-uncased --epochs 4 --batch-size 16 --lr 2e-5

This creates `models/distilbert_base_uncased/` (`model.pt`, tokenizer files, `meta.json`).
Setup instructions: `docs/DL_SETUP_MAC.md`. Full results: `dl/results/distilbert_base_uncased/`.

## smoke_random_init/ (testing only, if present)

A tiny, randomly-initialised model with the DistilBERT architecture, used only to check that the code runs without internet access. It is **not** transfer learning and must not be reported as a DistilBERT result.