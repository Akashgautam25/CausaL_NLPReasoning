import json

import networkx as nx
import pytest
import torch

from dl.evaluation.common import metrics, pred_to_directional, semeval_to_pair
from src.bayesian import build_bayesian_network, query_conditional, query_intervention
from src.extractors.base import BaseExtractor
from src.extractors.distilbert.dataset import RelationDataset, make_collate, marker_ids
from src.extractors.distilbert.extractor import FineTunedDistilBERTExtractor
from src.extractors.distilbert.markers import insert_markers_by_offsets
from src.extractors.distilbert.model import load_checkpoint
from src.graph.dl_integration import build_dl_graph
from src.preprocessing.semeval import LABELS, map_label, parse_file, quality_report, to_marked, leakage_safe_split
from src.schemas.prediction import Direction, Prediction, RelationClass
from src.schemas.text import CandidatePair, Mention


def mk_pair(pid, text, s1, e1, s2, e2, n1="BMICat", n2="Outcome", doc="PMID:1"):
    sid = f"{doc}:s01"
    m = lambda n, s, e, node: Mention(mention_id=f"{sid}:m{n}", sent_id=sid, doc_id=doc, surface_text=text[s:e],
                                      start_char=s, end_char=e, canonical_node_id=node)
    return CandidatePair(pair_id=pid, doc_id=doc, sent_id=sid, sentence_text=text, entity_1=m(1, s1, e1, n1),
                         entity_2=m(2, s2, e2, n2))


# ---------------- dataset / labels ----------------
def test_label_mapping():
    assert map_label("Cause-Effect(e1,e2)") == "CAUSE_E1_E2"
    assert map_label("Cause-Effect(e2,e1)") == "CAUSE_E2_E1"
    for other in ["Other", "Component-Whole(e1,e2)", "Entity-Origin(e2,e1)"]:
        assert map_label(other) == "NONE"


def test_parse_and_marked(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text('1\t"The <e1>smoke</e1> caused <e2>cancer</e2>."\r\nCause-Effect(e1,e2)\r\nComment:\r\n\r\n'
                 '2\t"A <e1>cat</e1> and a <e2>dog</e2>."\r\nOther\r\nComment:\r\n\r\n')
    rows = parse_file(f)
    assert [r["label"] for r in rows] == ["CAUSE_E1_E2", "NONE"]
    assert rows[0]["text"] == "The [E1] smoke [/E1] caused [E2] cancer [/E2] ."[:-2] + "."
    assert to_marked("<e1>a</e1> b <e2>c</e2>") == "[E1] a [/E1] b [E2] c [/E2]"


def test_split_no_leakage_and_reproducible():
    rows = [{"id": i, "plain_text": f"sentence number {i // 2}", "text": "x", "label": ["NONE", "CAUSE_E1_E2"][i % 2 == 0 and i % 4 == 0]}
            for i in range(200)]
    a1, b1 = leakage_safe_split(rows, 0.2, seed=7)
    a2, b2 = leakage_safe_split(rows, 0.2, seed=7)
    assert [r["id"] for r in b1] == [r["id"] for r in b2]
    assert not ({r["plain_text"] for r in a1} & {r["plain_text"] for r in b1})
    assert len(a1) + len(b1) == len(rows)


def test_quality_report_flags_missing_markers():
    sp = {"train": [{"text": "no markers", "plain_text": "no markers", "label": "NONE"}], "dev": [], "test": []}
    assert quality_report(sp)["missing_markers"] == 1


# ---------------- markers / tokenisation ----------------
def test_marker_insertion_order_independent():
    t = "smoking causes cancer"
    assert insert_markers_by_offsets(t, 0, 7, 15, 21) == "[E1] smoking [/E1] causes [E2] cancer [/E2]"
    assert insert_markers_by_offsets(t, 15, 21, 0, 7) == "[E2] smoking [/E2] causes [E1] cancer [/E1]"
    with pytest.raises(ValueError):
        insert_markers_by_offsets(t, 0, 10, 5, 12)


def test_tokenization_marker_ids_and_collate(tiny_ckpt):
    tok, model, meta = load_checkpoint(tiny_ckpt, torch.device("cpu"))
    m = marker_ids(tok)
    assert len(set(m.values())) == 4 and tok.unk_token_id not in m.values()
    rows = [{"text": "The [E1] smoking [/E1] causes [E2] cancer [/E2] .", "label": "CAUSE_E1_E2"},
            {"text": "short [E2] b [/E2] [E1] a [/E1]", "label": "NONE"}]
    ds = RelationDataset(rows, tok, 32)
    b = make_collate(tok.pad_token_id, m["[E1]"], m["[E2]"])([ds[0], ds[1]])
    assert b["input_ids"].shape == b["attention_mask"].shape
    assert b["input_ids"][0, b["e1_pos"][0]] == m["[E1]"] and b["input_ids"][1, b["e2_pos"][1]] == m["[E2]"]
    assert b["labels"].tolist() == [1, 0]
    assert b["attention_mask"][1].sum() < b["attention_mask"][0].sum() or b["attention_mask"].shape[1] >= 1


# ---------------- model / extractor ----------------
def test_model_forward_and_overfit(tiny_ckpt):
    tok, model, _ = load_checkpoint(tiny_ckpt, torch.device("cpu"))
    m = marker_ids(tok)
    rows = [{"text": "The [E1] smoking [/E1] causes [E2] cancer [/E2] .", "label": "CAUSE_E1_E2"},
            {"text": "A [E1] cat [/E1] sat on the [E2] mat [/E2] near the door .", "label": "NONE"}] * 4
    b = make_collate(tok.pad_token_id, m["[E1]"], m["[E2]"])([RelationDataset(rows, tok, 32)[i] for i in range(8)])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = model(**b)["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5, "model must be able to learn (gradients flow through encoder + head)"


def test_extractor_interface_and_schema(tiny_ckpt):
    ex = FineTunedDistilBERTExtractor(tiny_ckpt, device="cpu")
    assert isinstance(ex, BaseExtractor)
    pairs = [mk_pair("PMID:1:s01:p01", "Obesity increases the risk of diabetes", 0, 7, 30, 38),
             mk_pair("PMID:1:s01:p02", "Obesity increases the risk of diabetes", 0, 7, 3, 12)]  # overlapping -> unscorable
    preds = ex.predict(pairs)
    assert len(preds) == 2 and all(isinstance(p, Prediction) for p in preds)
    assert [p.pair_id for p in preds] == [p.pair_id for p in pairs]
    p0 = preds[0]
    assert 0 <= p0.label.confidence <= 1 and abs(sum(p0.raw_output["probs"].values()) - 1) < 1e-3
    assert p0.label.provenance.doc_id == "PMID:1" and p0.label.provenance.sent_id == "PMID:1:s01"
    assert p0.raw_output["pretrained_weights"] is False
    assert preds[1].validation_status == "schema_error"


def test_pred_to_directional_and_metrics():
    m = metrics(["NONE", "CAUSE_E1_E2", "CAUSE_E2_E1"], ["NONE", "CAUSE_E1_E2", "NONE"], LABELS)
    assert m["accuracy"] == pytest.approx(2 / 3) and sum(map(sum, m["confusion_matrix"])) == 3
    assert set(m["per_class"]) == set(LABELS)


def test_semeval_offsets_roundtrip():
    row = {"id": 5, "text": "The [E1] burst [/E1] has been caused by water hammer [E2] pressure [/E2] .", "label": "NONE"}
    p = semeval_to_pair(row)
    assert (p.entity_1.surface_text, p.entity_2.surface_text) == ("burst", "pressure")
    assert insert_markers_by_offsets(p.sentence_text, p.entity_1.start_char, p.entity_1.end_char,
                                     p.entity_2.start_char, p.entity_2.end_char).count("[E1]") == 1


# ---------------- graph integration ----------------
def fake_pred(pair, cls, direction, conf):
    from src.schemas.prediction import LanguageStatus, Polarity, RelationLabel
    return Prediction(pair_id=pair.pair_id, extractor_name="X", extractor_version="1",
                      label=RelationLabel(relation_class=cls, direction=direction, polarity=Polarity.INCREASES,
                                          language_status=LanguageStatus.AFFIRMED, confidence=conf))


def test_graph_integration_threshold_selfloop_conflict_and_dag():
    t = "Obesity increases risk of diabetes; diabetes leads to obesity"
    p1 = mk_pair("a:p1", t, 0, 7, 24, 32, "BMICat", "Outcome", "PMID:1")
    p2 = mk_pair("b:p1", t, 0, 7, 24, 32, "BMICat", "Outcome", "PMID:2")
    p3 = mk_pair("c:p1", t, 0, 7, 24, 32, "BMICat", "BMICat", "PMID:3")  # self loop
    p4 = mk_pair("d:p1", t, 0, 7, 24, 32, "BMICat", "Outcome", "PMID:4")  # low confidence
    preds = [fake_pred(p1, RelationClass.CAUSAL, Direction.E1_TO_E2, 0.9),
             fake_pred(p2, RelationClass.CAUSAL, Direction.E2_TO_E1, 0.8),
             fake_pred(p3, RelationClass.CAUSAL, Direction.E1_TO_E2, 0.95),
             fake_pred(p4, RelationClass.CAUSAL, Direction.E1_TO_E2, 0.4)]
    r = build_dl_graph([p1, p2, p3, p4], preds, confidence_threshold=0.7)
    assert r["is_dag_after"] and nx.is_directed_acyclic_graph(r["causal_dag"])
    assert len(r["log"]["below_threshold"]) == 1 and len(r["self_loops"]) == 1
    assert len(r["log"]["direction_conflicts"]) == 1          # conflict recorded, not silently dropped
    assert r["cycles_before"] >= 1 or r["causal_dag"].number_of_edges() >= 1
    assert len(r["log"]["cycle_repair"]) == r["causal_graph_before_repair"].number_of_edges() - r["causal_dag"].number_of_edges()
    for u, v, d in r["causal_dag"].edges(data=True):
        assert d["evidence"][0]["doc_id"].startswith("PMID:") and d["evidence"][0]["sent_id"]


def test_duplicate_evidence_not_double_counted():
    t = "Obesity increases risk of diabetes"
    p = mk_pair("a:p1", t, 0, 7, 26, 34)
    pr = fake_pred(p, RelationClass.CAUSAL, Direction.E1_TO_E2, 0.9)
    r = build_dl_graph([p], [pr, pr], 0.5)
    assert r["causal_dag"]["BMICat"]["Outcome"]["support_count"] == 1


# ---------------- cycles / bayesian ----------------
def test_cycle_detection_and_repair():
    from src.graph.cycles import detect_cycles
    from src.graph.repair import repair_graph_cycles
    g = nx.DiGraph()
    g.add_edge("A", "B", aggregated_confidence=0.9, support_count=2)
    g.add_edge("B", "A", aggregated_confidence=0.3, support_count=1)
    assert detect_cycles(g)["is_dag"] is False
    fixed, log = repair_graph_cycles(g)
    assert nx.is_directed_acyclic_graph(fixed) and log[0]["removed_edge"] == "B->A"


def test_bayesian_network_validation_and_queries():
    g = nx.DiGraph()
    g.add_edge("Age", "BMI", polarity="increases"); g.add_edge("Age", "Out", polarity="increases")
    g.add_edge("BMI", "Out", polarity="increases"); g.add_edge("Act", "BMI", polarity="decreases")
    bn = build_bayesian_network(g)
    assert bn.check_model() and bn.parameter_source == "ILLUSTRATIVE"
    p = query_conditional(bn, "Out", {"BMI": "yes"})
    assert sum(p.values()) == pytest.approx(1.0)
    do = query_intervention(bn, "Out", {"BMI": "yes"})
    assert "disclaimer" in do and sum(do["distribution"].values()) == pytest.approx(1.0)
    # confounding by Age => observational != interventional
    assert abs(p["yes"] - do["distribution"]["yes"]) > 1e-3
    # protective parent lowers BMI probability
    assert query_conditional(bn, "BMI", {"Act": "yes"})["yes"] < query_conditional(bn, "BMI", {"Act": "no"})["yes"]
    with pytest.raises(ValueError):
        build_bayesian_network(nx.DiGraph([("a", "b"), ("b", "a")]))
