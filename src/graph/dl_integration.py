"""
Integration of transformer predictions into the existing evidence-graph pipeline.

Policy (all decisions are logged, nothing is dropped silently):
  1. Predictions below `confidence_threshold`, unscorable pairs (schema_error) -> `below_threshold` / `unscorable` logs.
  2. NONE predictions create no edge.
  3. The existing EvidenceGraphBuilder keeps handling: canonicalisation, self-loop rejection (logged in
     `filtered_self_loops`), duplicate evidence (one evidence item per sentence), polarity flips, and the
     epidemiological routing rules. Doc ids / sentence ids / char spans travel with every evidence item.
  4. Direction conflicts (A->B and B->A both asserted) are recorded in `direction_conflicts` BEFORE repair.
  5. Remaining cycles are removed by the existing deterministic min-confidence repair; every removal is logged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from src.graph.builder import EvidenceGraphBuilder, resolve_canonical_node
from src.graph.cycles import detect_cycles
from src.graph.repair import repair_graph_cycles
from src.schemas.prediction import Direction, Prediction, RelationClass
from src.schemas.text import CandidatePair


def build_dl_graph(pairs: list[CandidatePair], predictions: list[Prediction], confidence_threshold: float = 0.7,
                   extra_predictions: list[Prediction] | None = None) -> dict[str, Any]:
    pair_by_id = {p.pair_id: p for p in pairs}
    log: dict[str, Any] = {"threshold": confidence_threshold, "below_threshold": [], "unscorable": [],
                           "direction_conflicts": [], "cycle_repair": []}
    kept: list[Prediction] = []
    for pred in predictions:
        if pred.validation_status != "valid":
            log["unscorable"].append({"pair_id": pred.pair_id, "reason": pred.error_message})
            continue
        if pred.label.relation_class == RelationClass.NONE:
            continue
        if pred.label.confidence < confidence_threshold:
            log["below_threshold"].append({"pair_id": pred.pair_id, "confidence": pred.label.confidence,
                                           "class": pred.label.relation_class.value})
            continue
        kept.append(pred)
    # associations from other extractors (e.g. spaCy rules) can be merged: the DL model never emits them
    all_preds = kept + [p for p in (extra_predictions or []) if p.label.relation_class != RelationClass.NONE]

    builder = EvidenceGraphBuilder()
    builder.build_from_pairs_and_predictions(pairs, all_preds)

    # direction conflicts among DL causal claims (by canonical node pair)
    claims: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pred in kept:
        pr = pair_by_id.get(pred.pair_id)
        if pr is None or pred.label.relation_class != RelationClass.CAUSAL:
            continue
        a = resolve_canonical_node(pr.entity_1.canonical_node_id, pr.entity_1.surface_text)
        b = resolve_canonical_node(pr.entity_2.canonical_node_id, pr.entity_2.surface_text)
        if not a or not b or a == b:
            continue
        s, t = (a, b) if pred.label.direction == Direction.E1_TO_E2 else (b, a)
        claims[(s, t)].append(pred.pair_id)
    for (s, t), ids in claims.items():
        if s < t and (t, s) in claims:
            log["direction_conflicts"].append({"edge_a": f"{s}->{t}", "support_a": claims[(s, t)],
                                               "edge_b": f"{t}->{s}", "support_b": claims[(t, s)]})

    causal = builder.causal_graph
    before = detect_cycles(causal)
    repaired, repair_log = repair_graph_cycles(causal)
    log["cycle_repair"] = repair_log
    after = detect_cycles(repaired)
    assert after["is_dag"], "repair must yield a DAG"
    return {"causal_graph_before_repair": causal, "causal_dag": repaired,
            "association_graph": builder.association_graph, "filtering": builder.get_filtering_stats(),
            "self_loops": builder.filtered_self_loops, "cycles_before": before["cycle_count"],
            "is_dag_after": after["is_dag"], "log": log}
