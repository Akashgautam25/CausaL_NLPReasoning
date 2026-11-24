"""Shared metric helpers and SemEval -> CandidatePair conversion."""
from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)

from src.schemas.prediction import Direction, Prediction, RelationClass
from src.schemas.text import CandidatePair, Mention


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "n": len(y_true), "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def semeval_to_pair(row: dict, idx: int = 0) -> CandidatePair:
    """Rebuild plain text and char offsets of the two marked nominals from a marked SemEval row."""
    return _pair_from_marked(row, idx)


def _pair_from_marked(row: dict, idx: int) -> CandidatePair:
    parts = re.split(r"\[(/?)E([12])\]", row["text"])
    # parts: text, slash, num, text, slash, num, ...
    i = 1
    starts, ends = {}, {}
    text = ""
    text += parts[0]
    while i < len(parts):
        slash, num, seg = parts[i], parts[i + 1], parts[i + 2]
        seg_stripped = seg
        if slash == "":
            seg_stripped = seg.lstrip(" ")
            starts[num] = len(text)
        else:
            text = text.rstrip(" ")
            ends[num] = len(text)
        text += seg_stripped
        i += 3
    text = text.strip()
    sid = f"SEMEVAL:{row['id']}:s0"
    def m(n):
        return Mention(mention_id=f"{sid}:m{n}", sent_id=sid, doc_id=f"SEMEVAL:{row['id']}",
                       surface_text=text[starts[n]:ends[n]], start_char=starts[n], end_char=ends[n])
    return CandidatePair(pair_id=f"SEMEVAL:{row['id']}:p01", doc_id=f"SEMEVAL:{row['id']}", sent_id=sid,
                         sentence_text=text, entity_1=m("1"), entity_2=m("2"), split="test")


def pred_to_directional(p: Prediction) -> str:
    """Project a schema Prediction to the 3-way directional label (association_only -> NONE)."""
    l = p.label
    if l.relation_class == RelationClass.CAUSAL:
        if l.direction == Direction.E1_TO_E2:
            return "CAUSE_E1_E2"
        if l.direction == Direction.E2_TO_E1:
            return "CAUSE_E2_E1"
        return "CAUSE_E1_E2"
    return "NONE"
