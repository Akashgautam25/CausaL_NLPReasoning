# models/

* `smoke_random_init/` — checkpoint from the **offline pipeline-verification run** (small, randomly-initialised, DistilBERT *architecture*, vocab trained on SemEval train text). It exists so tests/notebook/graph-integration run without internet. **It is not transfer learning and its scores must not be reported as DistilBERT results.**
* `distilbert_base_uncased/` — created when YOU run `python -m dl.training.train` on a machine that can download `distilbert-base-uncased` (see `docs/DL_SETUP_MAC.md`). Not included in this ZIP because it could not be trained in the build environment (no Hugging Face access).
