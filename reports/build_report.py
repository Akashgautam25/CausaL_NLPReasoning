"""Build reports/DL_Project_Report.pdf from the results on disk. Re-run after training on the Mac:  python reports/build_report.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "dl/results"
TAG = "distilbert_base_uncased" if (RES / "distilbert_base_uncased/comparison_results.json").exists() else "smoke_random_init"
PRE = TAG == "distilbert_base_uncased"
cmp_ = json.loads((RES / TAG / "comparison_results.json").read_text())
trn = json.loads((RES / f"training_summary_{TAG}.json").read_text())
gint = json.loads((RES / f"graph_{TAG}/integration_summary.json").read_text())
bay = json.loads((RES / "bayesian/bayesian_demo_results.json").read_text())
audit = json.loads((ROOT / "data/processed/semeval_quality_report.json").read_text())

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, alignment=4)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=12)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8)
SM = ParagraphStyle("SM", parent=B, fontSize=8.5, leading=11)
WARN = ParagraphStyle("W", parent=B, backColor=colors.HexColor("#fff1cc"), borderColor=colors.HexColor("#e0a800"),
                      borderWidth=1, borderPadding=6, spaceBefore=6, spaceAfter=8)
story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
def H(t, s=H1): story.append(Paragraph(t, s))
def img(path, w=11.5):
    p = Path(path)
    if p.exists():
        from PIL import Image as PI
        iw, ih = PI.open(p).size
        story.append(Image(str(p), width=w * cm, height=w * cm * ih / iw))
def table(rows, widths=None, font=8.5):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde7f5")), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), font), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t); story.append(Spacer(1, 6))

A = cmp_["gold_semeval"]
LAB = ["NONE", "CAUSE_E1_E2", "CAUSE_E2_E1"]
best = max(A, key=lambda k: A[k]["macro_f1"])
f = lambda x: f"{x:.3f}"

# ---- training curve figure ----
h = trn["history"]
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ax[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], "o-", label="train")
ax[0].plot([r["epoch"] for r in h], [r["val_loss"] for r in h], "s-", label="validation"); ax[0].legend(); ax[0].set_title("Loss"); ax[0].set_xlabel("epoch")
ax[1].plot([r["epoch"] for r in h], [r["val_macro_f1"] for r in h], "o-g"); ax[1].set_title("Validation macro-F1"); ax[1].set_xlabel("epoch")
fig.tight_layout(); curve = RES / TAG / "training_curves.png"; fig.savefig(curve, dpi=140); plt.close(fig)

# ================= title + abstract =================
story.append(Spacer(1, 2 * cm))
P("<b>Neuro-Symbolic Causal Inference Using Transformer-Based Relation Extraction and Bayesian Reasoning</b>", ParagraphStyle("T", parent=ss["Title"], fontSize=19, leading=24))
P("Deep Learning extension of: <i>Neuro-Symbolic Causal Inference via Language Model Knowledge Extraction and Bayesian Networks</i>", ParagraphStyle("S", parent=B, alignment=1))
P("B.Tech Computer Science - Deep Learning project report", ParagraphStyle("S2", parent=B, alignment=1))
if not PRE:
    P("<b>STATUS OF THE NUMBERS IN THIS PDF.</b> This copy was generated from the <b>offline smoke run</b>: a small, randomly-initialised model with the DistilBERT architecture, "
      "because pretrained <i>distilbert-base-uncased</i> could not be downloaded in the build environment. These results verify the pipeline only and are <b>not</b> "
      "transfer-learning results. Run the training step on the target machine and re-run <font face='Courier'>python reports/build_report.py</font> to regenerate this report with the real fine-tuned numbers.", WARN)
else:
    P("<b>Numbers in this report come from fine-tuned pretrained distilbert-base-uncased</b> (see Section 11 for device, time and seed).", WARN)
H("Abstract")
P("The preceding Applied Machine Learning project extracted candidate cause-effect relations from diabetes abstracts with rule-based, keyword and prompted-LLM extractors and assembled them into an evidence graph. "
  "Its audit showed that the 'DistilBERT' extractor was a keyword heuristic and that its 250 labels were rule-generated (4 causal, none in validation/test). "
  "This work adds a genuinely fine-tunable DistilBERT relation classifier with entity markers and entity-start pooling, trained on the human-annotated SemEval-2010 Task 8 corpus "
  "(Cause-Effect with direction mapped to CAUSE_E1_E2 / CAUSE_E2_E1 / NONE). Predictions are exposed through the existing extractor interface, filtered by confidence, merged into the provenance-preserving evidence graph with logged conflict and cycle handling, "
  "and passed to a pgmpy Bayesian network whose assumptions are made explicit. "
  f"On the SemEval gold test set ({A['spaCy-Rules']['n']} sentences) the highest macro-F1 was obtained by <b>{best}</b> ({f(A[best]['macro_f1'])}).")
story.append(PageBreak())

H("1. Introduction")
P("Scientific text states causal claims in many surface forms. Turning them into a machine-readable causal graph requires (i) reading comprehension to find the claim and its direction, and (ii) formal reasoning over the resulting graph. "
  "The project keeps these concerns separate: language models extract claims; graphs and Bayesian networks reason over them (neuro-symbolic design).")
H("2. Problem statement")
P("Given a sentence and two marked entity mentions, classify the asserted relation as NONE, CAUSE_E1_E2 or CAUSE_E2_E1; then aggregate accepted claims into an acyclic evidence graph and query a Bayesian network built on it.")
H("3. Objectives")
for t in ["Replace the misnamed heuristic with a genuine fine-tuned DistilBERT extractor, preserving the AML baseline.",
          "Train on human-annotated gold data with leakage-safe, reproducible splits; never present silver labels as ground truth.",
          "Compare AML and DL extractors on a common gold set and separately on silver data.",
          "Integrate with the existing graph pipeline (provenance, canonical nodes, self-loop rejection, duplicates, DAG).",
          "Provide a Bayesian network layer with honest causal-inference caveats."]:
    P("&bull; " + t)
H("4. Literature review")
P("The Transformer architecture replaced recurrence with self-attention (Vaswani et al., 2017). BERT pre-trains a deep bidirectional Transformer with masked-language modelling and is fine-tuned per task (Devlin et al., 2019). "
  "DistilBERT compresses BERT by knowledge distillation to 6 layers, reported by its authors as 40% smaller and 60% faster while retaining about 97% of BERT's language-understanding performance (Sanh et al., 2019). "
  "For relation classification, adding entity information helps: R-BERT inserts special characters around the entities and combines [CLS] with entity representations (Wu and He, 2019); Baldini Soares et al. (2019) found that entity-start-marker representations perform strongly. "
  "SemEval-2010 Task 8 (Hendrickx et al., 2010) is a standard benchmark of nominal relations including Cause-Effect. Corr2Cause (Jin et al., 2023) tests whether LLMs can infer causal-graph relations from correlation statements, which is a different task from sentence-level relation labelling, and BioRED (Luo et al., 2022) annotates biomedical relations that are association-style rather than explicitly causal. "
  "On the reasoning side, causal interpretation of graphs follows Pearl (2009), probabilistic graphical models follow Koller and Friedman (2009), and implementation uses pgmpy (Ankan and Panda, 2015).")
H("5. Existing AML methodology")
P("Europe PMC abstracts (500, frozen with SHA-256 hashes) are split into sentences, entities are mapped to canonical variables (e.g. BMICat, Outcome), and co-occurring pairs become <i>CandidatePair</i> objects (2,586 total, 1,143 non-self). "
  "Three extractors share one <i>BaseExtractor</i> contract: spaCy rules (dependency/trigger matching), a keyword heuristic, and prompted LLMs (Gemini/Groq, evaluated on a 20-pair pilot). "
  "An evidence graph separates directed causal edges from undirected associations, stores sentence-level provenance, and repairs cycles with a deterministic minimum-confidence rule.")
H("6. Proposed DL methodology")
P("<b>Input.</b> The sentence with entity markers: <font face='Courier'>... [E1] smoking [/E1] ... [E2] lung cancer [/E2] ...</font> The four markers are added to the tokenizer as special tokens and the embedding matrix is resized.")
P("<b>Model.</b> DistilBERT encoder (6 layers, hidden size 768, 12 heads). The classifier receives concat(h<sub>[CLS]</sub>, h<sub>[E1]</sub>, h<sub>[E2]</sub>) &rarr; dropout 0.1 &rarr; linear to 3 classes. Pooling at the markers lets the head know which entity is the candidate cause.")
P("<b>Training.</b> Class-weighted cross-entropy (inverse frequency, because ~87% of sentences are NONE), AdamW (lr 2e-5, weight decay 0.01), linear warm-up (10%) and decay, gradient clipping 1.0, batch size 16, max length 128, early stopping on validation macro-F1, fixed seed 42. "
  "Device selection: Apple MPS, then CUDA, then CPU.")
H("7. Dataset description")
P("Selected dataset: <b>SemEval-2010 Task 8</b>, human annotated. Mapping: Cause-Effect(e1,e2) &rarr; CAUSE_E1_E2; Cause-Effect(e2,e1) &rarr; CAUSE_E2_E1; every other relation (including Other) &rarr; NONE. SemEval has no association-only class, so the DL model does not learn it.")
sz, ld = audit["sizes"], audit["label_distribution"]
table([["split", "sentences"] + LAB] + [[s, sz[s]] + [ld[s].get(l, 0) for l in LAB] for s in ("train", "dev", "test")], [3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
P(f"Quality checks: empty texts {audit['empty_text']}, sentences missing a marker {audit['missing_markers']}, normalised-sentence overlap train/dev {audit['overlap_train_dev']} and train/test {audit['overlap_train_test']}; "
  f"{audit['test_dropped_for_leakage']} official test sentences were removed because they also occur in training data. SemEval has no document identifiers, so leakage control is at the sentence level (identical sentences stay in one split). Seed 42.", SM)
P("The original diabetes data (250 pairs) is retained as <b>silver</b>: rule-generated pseudo-labels, 4 causal / 85 association / 161 none, zero causal in dev and test. It is used only for the silver comparison. "
  "Corr2Cause and BioRED were assessed and not used: the former labels causal-inference validity rather than relations, the latter lacks an explicit causal label (their properties here are taken from the publications; only SemEval was downloaded and verified).")
H("8. DistilBERT architecture and implementation")
P("Code map: <i>src/preprocessing/semeval.py</i> (loading, mapping, splits), <i>src/extractors/distilbert/</i> (markers, dataset/collate, model, extractor), <i>dl/training/train.py</i> (loop, checkpoints), <i>dl/evaluation/</i> (metrics, comparison). "
  "The extractor implements <i>BaseExtractor.predict</i> and returns schema-validated <i>Prediction</i> objects with provenance (document id, sentence id, character span) and the softmax probability as confidence. "
  "Polarity and hedging are not modelled; they are set to explicit defaults and flagged in the raw output. Overlapping entity spans are returned as <i>schema_error</i> rather than guessed.")
H("9. Experimental setup")
P(f"Mode: <b>{cmp_['model_kind']}</b>. Device: {trn['device']}; torch {trn['torch']}. Training examples {trn['n_train']}, validation {trn['n_dev']}. "
  f"Epochs run: {len(h)} (early stopping), total training time {trn['total_training_seconds']:.1f} s. Best validation macro-F1: {f(trn['best_val_macro_f1'])}. Class weights: {[round(x, 2) for x in trn['class_weights']]}.")
img(curve, 14)
H("10. Results and analysis")
H("10.1 Gold evaluation: SemEval-2010 Task 8 test (human-annotated)", H2)
P("spaCy-Rules, KeywordHeuristic and DistilBERT were run on the identical sentences. Association predictions are projected to NONE because the gold data has no association class. 'Causal-vs-not F1' ignores direction.")
rows = [["extractor", "acc", "macro-P", "macro-R", "macro-F1", "wtd-F1", "F1 NONE", "F1 E1&rarr;E2", "F1 E2&rarr;E1", "causal-vs-not F1", "ms/sent"]]
for k, v in A.items():
    rows.append([k, f(v["accuracy"]), f(v["macro_precision"]), f(v["macro_recall"]), f(v["macro_f1"]), f(v["weighted_f1"])] + [f(v["per_class"][l]["f1"]) for l in LAB] + [f(v["binary_causal_f1"]), f"{v['inference_ms_per_sentence']:.2f}"])
rows[0] = [Paragraph(c, SM) for c in rows[0]]
table(rows, [3.1 * cm] + [1.45 * cm] * 10, 7.5)
for fn in ["gold_macro_f1.png", "gold_binary_f1.png"]:
    img(RES / TAG / fn, 9.5)
cmx = [RES / TAG / f"cm_gold_{k}.png" for k in A]
t = Table([[Image(str(p), width=5.2 * cm, height=4.5 * cm) for p in cmx]]); story.append(t)
P("Confusion matrices (gold test): rules, keyword heuristic, DistilBERT.", SM)
dist_ms = A["FineTuned-DistilBERT"]["inference_ms_per_sentence"]
if PRE:
    P(f"<b>Analysis.</b> {best} obtained the best macro-F1 ({f(A[best]['macro_f1'])}). Training took {trn['total_training_seconds']:.0f} s on {trn['device']}; DistilBERT inference cost {dist_ms:.2f} ms/sentence versus {A['spaCy-Rules']['inference_ms_per_sentence']:.2f} ms for the rules. "
      "Compare per-class F1 to see where each approach wins: rules are strong on sentences with explicit triggers and cost nothing to train, whereas the fine-tuned model can use context and entity roles for direction and for phrasings absent from the trigger list.")
else:
    P(f"<b>Analysis (smoke mode).</b> The randomly-initialised model reached macro-F1 {f(A['FineTuned-DistilBERT']['macro_f1'])}, far below spaCy-Rules ({f(A['spaCy-Rules']['macro_f1'])}). That is the expected outcome and the key lesson: "
      "with about 900 causal training sentences and no pretraining, a Transformer has no language knowledge to build on, so a hand-written trigger list wins. This is the viva answer to 'why can rule-based models beat deep learning on small data?'. "
      "It says nothing about pretrained DistilBERT, which must be measured by running the real training step.")
H("10.2 Silver evaluation: diabetes pairs (rule-generated labels, not ground truth)", H2)
S = cmp_["silver_diabetes"]["all_250_pairs"]
rows = [["extractor", "acc", "macro-F1", "F1 causal", "F1 assoc", "F1 none"]] + [[k, f(v["accuracy"]), f(v["macro_f1"])] + [f(v["per_class"][c]["f1"]) for c in ("causal", "association_only", "none")] for k, v in S.items()]
table(rows, [4.5 * cm] + [2.4 * cm] * 5)
P("Because the silver labels were created by regular-expression rules, spaCy-Rules scores are partly circular and must not be used to rank extractors. DistilBERT cannot score on association_only by construction. "
  "The saved Gemini/Groq predictions cover only 20 pairs, of which "
  f"{list(cmp_['silver_diabetes']['pilot_20_pairs'].values())[0]['n']} overlap the 250 silver pairs, so no LLM comparison is statistically meaningful; the LLM macro-F1 of about 0.26-0.27 in the AML audit table came from scoring the ~230 pairs the LLMs never saw as 'none', so it reflects missing coverage, not model quality.")
img(RES / TAG / "silver_macro_f1.png", 9.5)
H("10.3 Computational trade-offs", H2)
P(f"Rules and keyword heuristics need no training and run in about 0.1 ms/sentence on CPU. DistilBERT adds a training stage ({trn['total_training_seconds']:.0f} s in this run), about 66M parameters (for the pretrained model) and {dist_ms:.2f} ms/sentence inference here. "
  "LLM extractors need network calls, quota and money, are slower and non-deterministic, but need no task-specific training data.")
H("11. Graph construction")
P("Predictions below the confidence threshold are logged, not discarded silently. NONE creates no edge. The existing builder rejects self-loops after canonicalisation, merges duplicate evidence per sentence, keeps document/sentence provenance and applies the epidemiological routing rules. "
  "Opposite-direction claims between the same pair of nodes are written to a conflict log before the deterministic minimum-confidence cycle repair runs; every removed edge is logged and the final graph is asserted acyclic.")
table([["quantity", "value"], ["threshold", gint["threshold"]], ["candidate pairs", gint["n_pairs"]], ["predicted causal", gint["n_pred_causal"]], ["below threshold", gint["n_below_threshold"]],
       ["direction conflicts logged", gint["n_direction_conflicts"]], ["self-loops rejected", gint["self_loops_rejected"]], ["cycles before repair", gint["cycles_before_repair"]],
       ["edges removed by repair", gint["edges_removed_by_repair"]], ["final DAG nodes / edges", f"{gint['dag_nodes']} / {gint['dag_edges']}"], ["is DAG", gint["is_dag"]]], [6 * cm, 4 * cm])
if gint.get("WARNING"):
    P("<b>" + gint["WARNING"] + "</b>", WARN)
img(RES / f"graph_{TAG}/dl_causal_dag.png", 12)
H("12. Bayesian reasoning")
P("The acyclic graph is converted to a pgmpy DiscreteBayesianNetwork with binary variables. CPDs are either estimated from data (Bayesian estimator) or, when no suitable data exist, <b>illustrative</b> noisy-OR style numbers that are labelled as such. "
  "Model validity is checked with <i>check_model</i>; conditional queries use exact variable elimination and interventions use backdoor adjustment (pgmpy CausalInference).")
hb = bay["hypothetical_confounded_dag"]
k_obs = "P(Outcome=yes|BMICat=yes) [observational]"; k_do = "P(Outcome=yes|do(BMICat=yes)) [interventional, backdoor via Age]"
table([["query (hypothetical DAG, illustrative CPDs)", "value"], ["P(Outcome=yes)", f(hb["P(Outcome=yes)"])], ["P(Outcome=yes | BMI=yes) observational", f(hb[k_obs])],
       ["P(Outcome=yes | do(BMI=yes)) interventional", f(hb[k_do])], ["confounding gap", f(hb["confounding_gap"])]], [9 * cm, 3 * cm])
P("The gap arises because Age affects both BMI and the outcome; conditioning mixes association and causation, while do() removes the Age-to-BMI influence. The real extracted DAG contains a single edge (BMICat&rarr;Outcome), too sparse for meaningful intervention analysis. "
  "<b>Caveat:</b> " + bay["disclaimer"])
img(RES / "bayesian/bn_hypothetical_dag.png", 8)
H("13. Limitations")
for t in ["Domain shift: SemEval is general English, the application is biomedical; in-domain performance is unmeasured because no human-verified diabetes causal gold exists.",
          "No association-only class can be learned from SemEval; polarity and hedging are not modelled.",
          "A sentence that claims causation is not proof of causation; extraction from observational abstracts does not establish identification.",
          "Silver labels are rule-generated; the LLM pilot is too small to compare.",
          "Sentence-level only; entity pairs are limited by the candidate generator; confidences are uncalibrated softmax probabilities.",
          "Bayesian CPDs are illustrative; do-queries assume a correct DAG and no hidden confounders."]:
    P("&bull; " + t)
H("14. Conclusion")
P("A complete, tested pipeline now links leakage-safe gold data, an entity-aware fine-tunable DistilBERT, the existing provenance-preserving graph and a clearly caveated Bayesian layer. "
  "The scientifically meaningful comparison is the SemEval gold-test table produced with pretrained weights; until that run is done, conclusions about DistilBERT's advantage cannot be drawn. Future work: in-domain human annotation, biomedical encoders (BioBERT/PubMedBERT), polarity/hedging, calibration, and CPDs estimated from BRFSS/NHANES with proper adjustment.")
H("References")
refs = ["Vaswani, A. et al. (2017). Attention Is All You Need. <i>Advances in Neural Information Processing Systems 30</i>.",
        "Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. <i>NAACL-HLT</i>.",
        "Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.",
        "Hendrickx, I. et al. (2010). SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals. <i>Proceedings of the 5th International Workshop on Semantic Evaluation</i>, ACL.",
        "Wu, S., He, Y. (2019). Enriching Pre-trained Language Model with Entity Information for Relation Classification. <i>CIKM</i>.",
        "Baldini Soares, L., FitzGerald, N., Ling, J., Kwiatkowski, T. (2019). Matching the Blanks: Distributional Similarity for Relation Learning. <i>ACL</i>.",
        "Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing. <i>EMNLP: System Demonstrations</i>.",
        "Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. <i>NeurIPS 32</i>.",
        "Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. <i>ICLR</i>.",
        "Pearl, J. (2009). <i>Causality: Models, Reasoning, and Inference</i> (2nd ed.). Cambridge University Press.",
        "Koller, D., Friedman, N. (2009). <i>Probabilistic Graphical Models: Principles and Techniques</i>. MIT Press.",
        "Ankan, A., Panda, A. (2015). pgmpy: Probabilistic Graphical Models using Python. <i>Proceedings of the 14th Python in Science Conference (SciPy)</i>.",
        "Jin, Z. et al. (2023). Can Large Language Models Infer Causation from Correlation? arXiv:2306.05836 (Corr2Cause).",
        "Luo, L. et al. (2022). BioRED: a rich biomedical relation extraction dataset. <i>Briefings in Bioinformatics</i>, 23(5).",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12."]
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", SM)

out = ROOT / "reports/DL_Project_Report.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                  title="Neuro-Symbolic Causal Inference - DL Report").build(story)
print("wrote", out, "| mode:", TAG)
