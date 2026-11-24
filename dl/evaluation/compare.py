"""
AML vs DL comparison.

Part A (GOLD): SemEval-2010 Task 8 test split (human-annotated). spaCy-Rules, KeywordHeuristic and
               fine-tuned DistilBERT are all run live on the identical 2,700 sentences.
Part B (SILVER): 250 diabetes candidate pairs whose labels are rule-generated 'PROVISIONAL_RULE_PSEUDO_GOLD'
               (NOT human ground truth; spaCy-Rules is partly circular on it). Includes saved Gemini/Groq
               predictions, which only exist for a 20-pair pilot.

    python -m dl.evaluation.compare --checkpoint models/distilbert_base_uncased
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dl.evaluation.common import metrics, pred_to_directional, semeval_to_pair
from src.extractors.distilbert.extractor import FineTunedDistilBERTExtractor
from src.extractors.distilbert_extractor import KeywordHeuristicExtractor
from src.extractors.rules_extractor import RulesExtractor
from src.preprocessing.semeval import LABELS, load_split
from src.schemas.prediction import Prediction
from src.schemas.text import CandidatePair

ROOT = Path(__file__).resolve().parents[2]
SILVER = ["causal", "association_only", "none"]


def timed(ext, pairs):
    t0 = time.perf_counter()
    preds = ext.predict(pairs)
    dt = time.perf_counter() - t0
    return preds, dt


def plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black" if cm[i][j] < np.max(cm) / 2 else "white")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def bar(results: dict, key: str, title: str, path: Path, ylabel="score"):
    names = list(results)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    vals = [results[n][key] for n in names]
    ax.bar(names, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"][: len(names)])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    plt.xticks(rotation=15); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models/distilbert_base_uncased"))
    ap.add_argument("--tag", default=None, help="results sub-folder name (default: checkpoint dir name)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    ck = Path(a.checkpoint)
    tag = a.tag or ck.name
    out = ROOT / "dl/results" / tag
    out.mkdir(parents=True, exist_ok=True)

    db = FineTunedDistilBERTExtractor(ck, device=a.device)
    banner = "PRETRAINED DistilBERT fine-tuned" if db.pretrained else "RANDOM-INIT SMOKE MODEL (not transfer learning)"
    print("checkpoint:", banner)
    rules, kw = RulesExtractor(), KeywordHeuristicExtractor()
    results = {"checkpoint": str(ck), "model_kind": banner, "pretrained": db.pretrained,
               "device": str(db.device), "gold_semeval": {}, "silver_diabetes": {}}

    # ---------------- Part A: gold SemEval ----------------
    rows = load_split(ROOT / "data/processed", "test")
    pairs = [semeval_to_pair(r) for r in rows]
    y = [r["label"] for r in rows]
    A = {}
    for name, ext in [("spaCy-Rules", rules), ("KeywordHeuristic", kw), ("FineTuned-DistilBERT", db)]:
        preds, dt = timed(ext, pairs)
        m = metrics(y, [pred_to_directional(p) for p in preds], LABELS)
        yb = ["CAUSAL" if t != "NONE" else "NONE" for t in y]
        pb = ["CAUSAL" if pred_to_directional(p) != "NONE" else "NONE" for p in preds]
        m["binary_causal_f1"] = metrics(yb, pb, ["NONE", "CAUSAL"])["per_class"]["CAUSAL"]["f1"]
        m["inference_seconds_total"] = dt
        m["inference_ms_per_sentence"] = 1000 * dt / len(pairs)
        A[name] = m
        plot_cm(m["confusion_matrix"], LABELS, f"{name} (SemEval gold test)", out / f"cm_gold_{name}.png")
    tr = json.loads((ROOT / "dl/results" / f"training_summary_{'distilbert_base_uncased' if db.pretrained else 'smoke_random_init'}.json").read_text()) \
        if (ROOT / "dl/results" / f"training_summary_{'distilbert_base_uncased' if db.pretrained else 'smoke_random_init'}.json").exists() else {}
    A["FineTuned-DistilBERT"]["training_seconds"] = tr.get("total_training_seconds")
    results["gold_semeval"] = A
    bar({k: {"macro_f1": v["macro_f1"]} for k, v in A.items()}, "macro_f1", "Macro-F1 on SemEval GOLD test (3-way directional)", out / "gold_macro_f1.png", "macro-F1")
    bar({k: {"v": v["binary_causal_f1"]} for k, v in A.items()}, "v", "Causal-vs-not F1 on SemEval GOLD test", out / "gold_binary_f1.png", "F1")

    # ---------------- Part B: silver diabetes ----------------
    sample = [json.loads(l) for l in open(ROOT / "data/annotations/sample.jsonl")]
    sp = [CandidatePair.model_validate(r) for r in sample]
    gold = {r["pair_id"]: r["relation_class"] for r in sample}
    saved = {}
    for fn, key in [("spacy-rules", "spaCy-Rules"), ("keywordheuristic", "KeywordHeuristic"),
                    ("prompted-llm-gemini", "Gemini-LLM (20-pair pilot)"), ("prompted-llm-groq", "Groq-LLM (20-pair pilot)")]:
        f = ROOT / "results/predictions" / f"{fn}_predictions.jsonl"
        saved[key] = {p.pair_id: p for p in (Prediction.model_validate_json(l) for l in open(f))}
    dpreds, dt = timed(db, sp)
    saved["FineTuned-DistilBERT"] = {p.pair_id: p for p in dpreds}
    B = {}
    def cls(p): return p.label.relation_class.value
    for subset_name, ids in [("all_250_pairs", [r["pair_id"] for r in sample]),
                             ("pilot_20_pairs", sorted(set.intersection(*[set(v) for k, v in saved.items() if "LLM" in k]) & set(gold)))]:
        B[subset_name] = {}
        for name, pm in saved.items():
            use = [i for i in ids if i in pm]
            if len(use) != len(ids):
                continue
            m = metrics([gold[i] for i in use], [cls(pm[i]) for i in use], SILVER)
            B[subset_name][name] = m
            if subset_name == "all_250_pairs":
                plot_cm(m["confusion_matrix"], SILVER, f"{name} (diabetes SILVER)", out / f"cm_silver_{name}.png")
    B["distilbert_inference_ms_per_pair"] = 1000 * dt / len(sp)
    B["NOTE"] = ("Labels are rule-generated PROVISIONAL_RULE_PSEUDO_GOLD (not human-verified). Only 4/250 are causal; "
                 "test split has 0 causal. DistilBERT never predicts association_only (not in its training labels).")
    results["silver_diabetes"] = B
    bar({k: {"v": v["macro_f1"]} for k, v in B["all_250_pairs"].items()}, "v", "Macro-F1 on diabetes SILVER labels (250 pairs)", out / "silver_macro_f1.png", "macro-F1")

    # inference time chart
    names = list(A)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(names, [A[n]["inference_ms_per_sentence"] for n in names], color="#8172b3")
    ax.set_ylabel("ms / sentence"); ax.set_title("Inference time (SemEval test, this machine)", fontsize=10)
    plt.xticks(rotation=15); fig.tight_layout(); fig.savefig(out / "inference_time.png", dpi=140); plt.close(fig)

    (out / "comparison_results.json").write_text(json.dumps(results, indent=2))
    import csv
    with open(out / "comparison_table_gold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["extractor", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "binary_causal_f1"]
                   + [f"f1_{l}" for l in LABELS] + ["ms_per_sentence"])
        for n, m in A.items():
            w.writerow([n] + [round(m[k], 4) for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "binary_causal_f1"]]
                       + [round(m["per_class"][l]["f1"], 4) for l in LABELS] + [round(m["inference_ms_per_sentence"], 3)])
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ("accuracy", "macro_f1", "weighted_f1", "binary_causal_f1")} for k, v in A.items()}, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
