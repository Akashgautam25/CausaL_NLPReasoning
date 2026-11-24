# Setup & run — MacBook Pro M3 Pro (Apple Silicon, MPS)

```bash
# 1. Environment (Python 3.11 or 3.12, arm64)
cd CausaL_NLPReasoning
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dl.txt          # the macOS arm64 torch wheel already supports MPS
python -m spacy download en_core_web_sm

# 2. Check the accelerator
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# 3. Tests (no network or pretrained weights needed)
python -m pytest tests -q

# 4. Data (downloads SemEval-2010 Task 8 from GitHub, makes leakage-safe splits)
python -m src.preprocessing.semeval

# 5. THE REAL EXPERIMENT: fine-tune pretrained distilbert-base-uncased (needs internet once, ~270 MB)
python -m dl.training.train --model distilbert-base-uncased --epochs 4 --batch-size 16 --lr 2e-5
#    -> models/distilbert_base_uncased/ , dl/results/training_summary_distilbert_base_uncased.json
#    (device is auto-selected: MPS -> CUDA -> CPU; add --device cpu to force CPU.
#     If an MPS op is unsupported, PYTORCH_ENABLE_MPS_FALLBACK=1 is already set for you.)

# 6. AML vs DL comparison (gold + silver), charts, confusion matrices
python -m dl.evaluation.compare --checkpoint models/distilbert_base_uncased

# 7. Graph integration and Bayesian demo
python -m dl.run_graph_integration --checkpoint models/distilbert_base_uncased --threshold 0.7
python -m dl.run_bayesian_demo

# 8. Notebook (executes everything incl. training) and PDF report built from the real results
jupyter notebook dl/notebooks/DistilBERT_Causal_Relation_Extraction.ipynb
python reports/build_report.py
```
Typical cost: a few minutes per epoch on M3 Pro with batch 16 (not measured here — record your own number from `training_summary_*.json`).
