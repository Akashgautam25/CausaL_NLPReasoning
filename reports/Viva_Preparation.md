# Viva Preparation — Transformer-Based Causal Relation Extraction + Bayesian Reasoning

**Read this first (be honest in the viva).** Gold data = SemEval-2010 Task 8. The diabetes labels are rule-generated *silver* labels. The numbers in the repo's `smoke_random_init` folder come from a randomly-initialised mini-model used only to prove the pipeline runs — they are not DistilBERT results. Your real result is the one you produce by running the training step on your MacBook (see `docs/DL_SETUP_MAC.md`). Quote only those numbers, and say which device and seed produced them.

---

## A. The one-minute project summary
My AML project extracted cause-effect claims from diabetes abstracts using rules, keywords and LLMs and built a graph. Its audit showed the "DistilBERT" file was really a keyword heuristic and the labels were rule-made. For the DL project I built a real fine-tuned DistilBERT relation classifier trained on human-labelled SemEval-2010 Task 8 data (Cause-Effect with direction), plugged it into the same extractor interface and graph builder, compared it with the AML extractors, and added a pgmpy Bayesian network with clear caveats about causal validity.

---

## B. 30 viva questions

**1. What problem does the project solve?**
Given a sentence and two marked entities, decide whether the sentence says one *causes* the other (and which way), then collect these claims into a graph that can be used for probabilistic reasoning. Language understanding (neural) is separated from formal reasoning (graph + Bayesian network) — that separation is the "neuro-symbolic" idea.

**2. What is relation extraction?**
Finding semantic relations between entities in text, e.g. (smoking, lung cancer, causes). It is usually posed as classification: for each entity pair in a sentence, pick a relation label (or "none").

**3. What is causal inference, and is your system doing it?**
Causal inference estimates what would happen if we *intervene* on a variable, not just what co-occurs. My system does **not** establish causation; it *extracts causal claims made by authors*. Whether the claim is true needs study design, confounder control and data.

**4. Association vs causation?**
Association: two variables move together (ice-cream sales and drowning). Causation: changing one changes the other. Confounders (heat) create association without causation.

**5. What is transfer learning?**
Reusing knowledge learned on a huge general task (reading billions of words) for a small new task. Pretrained DistilBERT already knows grammar and word meaning; we only adapt it with ~7k labelled sentences.

**6. What is fine-tuning?**
Continuing training of a pretrained model on the target task, with a small learning rate (2e-5), updating all weights plus a new classification layer.

**7. Why DistilBERT instead of BERT?**
DistilBERT has 6 layers instead of 12, about 40% fewer parameters and is about 60% faster (Sanh et al., 2019), keeping most of BERT's accuracy. That fits a laptop (M3 Pro, MPS) and a student timeline. Trade-off: slightly lower accuracy than BERT.

**8. What is tokenization?**
Splitting text into units the model knows. WordPiece splits rare words into pieces ("hypertension" → "hyper", "##tension"). Each piece maps to an integer id. I add four special tokens `[E1] [/E1] [E2] [/E2]` that are never split.

**9. What are embeddings?**
Learned vectors (768 numbers in DistilBERT) representing tokens; similar meanings get nearby vectors. Position embeddings are added so the model knows word order. Contextual layers then change a token's vector depending on its sentence.

**10. What is attention?**
Each token computes how much to "look at" every other token: scores = (query · key) → softmax → weighted sum of values. It lets "causes" connect directly to both entities even if they are far apart. Multi-head attention does this several ways in parallel.

**11. Why entity markers?**
The same sentence can contain many entity pairs. Markers tell the model *which two* we are asking about and which is E1 vs E2 — essential for direction.

**12. What is your classification head?**
I concatenate three hidden vectors: `[CLS]` (sentence summary), the vector at `[E1]` and the vector at `[E2]`, apply dropout, then a linear layer to 3 classes: NONE, CAUSE_E1_E2, CAUSE_E2_E1. Entity-start pooling is used in R-BERT-style work and Matching-the-Blanks.

**13. Why SemEval-2010 Task 8, and how did you map labels?**
It is human-annotated, public and contains Cause-Effect with direction. Cause-Effect(e1,e2)→CAUSE_E1_E2, Cause-Effect(e2,e1)→CAUSE_E2_E1, everything else→NONE. I did not use Corr2Cause (it tests inferring causal-graph validity from correlation statements — a different task) or BioRED (relations are association-style, no explicit causal label).

**14. Why not just use your 250 diabetes examples?**
Only 4 are causal, none in validation/test, and the labels were produced by rules — training on them would teach the model to copy the rules, and evaluating on them is circular. They are kept only as clearly labelled *silver* data.

**15. How did you prevent data leakage?**
Official SemEval test kept separate; 17 test sentences that also appear in train removed; dev split carved from train with identical sentences kept together; zero normalised-sentence overlap verified in `semeval_quality_report.json`. SemEval has no document ids, so sentence-level is the best available unit. Fixed seed 42.

**16. How did you handle class imbalance?**
About 87% of sentences are NONE. I used inverse-frequency class weights in the cross-entropy loss and judged models by macro-F1 (each class counts equally), not accuracy.

**17. Explain precision, recall, F1, macro vs weighted F1.**
Precision = of the claims predicted causal, how many were right. Recall = of the truly causal ones, how many were found. F1 = harmonic mean, punishes imbalance between the two. Macro-F1 averages classes equally (good for rare classes); weighted-F1 averages by class size (dominated by NONE). Accuracy alone is misleading: predicting NONE always gives ~87%.

**18. What do the confusion matrix and per-class F1 tell you?**
Rows = true class, columns = predicted. Off-diagonal cells show which mistakes happen, e.g. direction flips (E1→E2 predicted as E2→E1) versus missed relations (causal predicted NONE).

**19. Training details?**
AdamW, lr 2e-5, weight decay 0.01, batch 16, 4 epochs, max length 128, warm-up 10% then linear decay, gradient clipping, early stopping on dev macro-F1, best checkpoint saved. Device MPS → CUDA → CPU.

**20. How do you know the model is not overfitting?**
Compare train and validation loss per epoch; keep the checkpoint with best dev macro-F1; report final numbers on a test set never used for choosing anything.

**21. Why can rule-based models beat deep learning on small data?**
Rules encode expert knowledge for free ("caused by", "leads to"); a neural network must learn language from examples. With few examples (especially few causal ones) or without pretraining it cannot. In my own offline smoke run, a randomly-initialised model got far below spaCy-Rules — pretraining is what closes that gap. Rules also win on speed, determinism and interpretability.

**22. Where should DistilBERT beat rules?**
Phrasings not in the trigger list, long-range structure, using context to decide direction, and negation-like cues. Where rules win: sentences with explicit triggers, zero training cost, 0.1 ms/sentence, fully explainable. Check your per-class F1 table to say which actually happened in *your* run.

**23. Why is your DistilBERT evaluated on SemEval and not on diabetes data?**
Because SemEval is the only human-annotated data I have. That means domain shift (general English vs biomedical) is a real limitation; in-domain performance is unmeasured until someone annotates diabetes sentences. The repo already includes a 50-pair annotation queue for that.

**24. What is a DAG and why must the graph be acyclic?**
Directed Acyclic Graph: arrows with no path returning to its start. A Bayesian network factorises the joint distribution as a product of "node given its parents"; a cycle would make that definition circular (A depends on B depends on A). Causally, a cycle means A causes B causes A, which static models cannot represent (time-unrolled models can).

**25. How did you make the graph acyclic without hiding information?**
Direction conflicts (A→B and B→A) are logged first; then the existing deterministic min-confidence repair removes the weakest edge in each cycle, and every removal is logged. Self-loops, below-threshold and unscorable pairs are also logged — nothing is dropped silently.

**26. What is a Bayesian network?**
A DAG plus a conditional probability table (CPD) for each node given its parents. It lets us compute P(outcome | evidence) with exact inference (variable elimination).

**27. How are your CPDs obtained? Are they real?**
When no suitable dataset is attached, they are *illustrative* noisy-OR style numbers and the code labels them `ILLUSTRATIVE`. If you supply a data table (e.g. BRFSS), they are estimated (`ESTIMATED`). I never present illustrative numbers as learned results.

**28. P(Y | X) vs P(Y | do(X))?**
P(Y|X): among people who *have* X, how common is Y (includes confounding). P(Y|do(X)): if we *force* X, how common is Y. In my demo, Age affects both BMI and outcome, so the two differ (gap shown in the report). do() answers need a correct DAG and no hidden confounders.

**29. Can your system claim that BMI causes diabetes?**
No. It reports that sentences *assert* it and, given an assumed DAG, what the model implies. Text extraction from observational abstracts does not give causal identification. The real extracted DAG has one edge, too sparse for meaningful intervention analysis.

**30. What are the limitations of your approach?**
Domain shift (SemEval vs biomedical); no association class in SemEval; polarity/hedging not modelled; sentence-level only; uncalibrated softmax confidences; silver labels for diabetes; LLM comparison limited to a 20-pair pilot (only a couple overlap with silver labels); illustrative CPDs; identification assumptions for do(). Future work: annotate in-domain data, BioBERT/PubMedBERT, calibration, real CPDs.

---

## C. Quick-fire facts
* DistilBERT-base-uncased: 6 layers, 768 hidden, 12 heads, ~66M params, lower-cased vocabulary of ~30k WordPieces.
* Output classes: NONE / CAUSE_E1_E2 / CAUSE_E2_E1. It never predicts "association" (not in training data).
* Confidence = softmax probability of predicted class; graph threshold default 0.7.
* Run: `python -m dl.training.train` → `python -m dl.evaluation.compare` → `python -m dl.run_graph_integration` → `python reports/build_report.py`.
* Tests: `python -m pytest tests -q` (82 passed in the build environment).

## D. Questions examiners may trap you with
* *"What is your accuracy?"* — Give macro-F1 and per-class F1 from **your** run, and say accuracy is inflated by the NONE class.
* *"Did DistilBERT beat the rules?"* — Answer from your table only; if not, explain small data / domain shift / explicit triggers.
* *"Is the graph the truth?"* — No: it is a map of what the literature claims.
* *"Why are the LLM results missing?"* — API predictions exist for only 20 pairs; evaluation on so few is meaningless, so they are reported but not ranked.
