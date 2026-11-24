"""
Run a fine-tuned DistilBERT checkpoint over the diabetes candidate pairs and build the evidence DAG.

    python -m dl.run_graph_integration --checkpoint models/distilbert_base_uncased --threshold 0.7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.extractors.distilbert.extractor import FineTunedDistilBERTExtractor
from src.graph.dl_integration import build_dl_graph
from src.graph.exporter import export_graph_for_phase2
from src.graph.visualizer import plot_evidence_graph
from src.schemas.prediction import Prediction
from src.schemas.text import CandidatePair

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models/distilbert_base_uncased"))
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--with-rule-associations", action="store_true",
                    help="also merge spaCy-Rules association edges (the DL model never predicts association)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    ck = Path(a.checkpoint)
    out = ROOT / "dl/results" / f"graph_{ck.name}"
    out.mkdir(parents=True, exist_ok=True)

    pairs = [CandidatePair.model_validate_json(l) for l in open(ROOT / "data/candidates_corpus.jsonl")]
    ex = FineTunedDistilBERTExtractor(ck, device=a.device)
    preds = ex.predict(pairs)
    extra = None
    if a.with_rule_associations:
        extra = [p for p in (Prediction.model_validate_json(l) for l in open(ROOT / "results/predictions/spacy-rules_predictions.jsonl"))
                 if p.label.relation_class.value == "association_only"]
    r = build_dl_graph(pairs, preds, a.threshold, extra)
    dag = r["causal_dag"]
    export_graph_for_phase2(dag, out / "dl_causal_dag.graphml", out / "dl_provenance.json",
                            repair_log=r["log"]["cycle_repair"], association_graph=r["association_graph"])
    if dag.number_of_edges():
        plot_evidence_graph(dag, out / "dl_causal_dag.png", title="DistilBERT-derived causal DAG" + ("" if ex.pretrained else " [RANDOM-INIT SMOKE MODEL - mechanics demo only]"))
    summary = {"checkpoint": str(ck), "pretrained_weights": ex.pretrained, "threshold": a.threshold,
               "n_pairs": len(pairs), "n_pred_causal": sum(p.label.relation_class.value == "causal" for p in preds),
               "n_unscorable": len(r["log"]["unscorable"]), "n_below_threshold": len(r["log"]["below_threshold"]),
               "n_direction_conflicts": len(r["log"]["direction_conflicts"]), "self_loops_rejected": len(r["self_loops"]),
               "cycles_before_repair": r["cycles_before"], "edges_removed_by_repair": len(r["log"]["cycle_repair"]),
               "dag_nodes": dag.number_of_nodes(), "dag_edges": dag.number_of_edges(), "is_dag": r["is_dag_after"],
               "edges": [f"{u}->{v}" for u, v in dag.edges()],
               "WARNING": None if ex.pretrained else "random-init smoke checkpoint: mechanics demo only, edges are NOT scientific findings"}
    (out / "integration_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "integration_log.json").write_text(json.dumps(r["log"], indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
