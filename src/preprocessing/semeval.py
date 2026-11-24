"""
SemEval-2010 Task 8 loading, label mapping, and leakage-safe splitting.

Source (verified): Hendrickx et al., 2010, "SemEval-2010 Task 8: Multi-Way Classification of
Semantic Relations Between Pairs of Nominals". 8,000 training and 2,717 test sentences, each with
two marked nominals <e1>..</e1>, <e2>..</e2> and a HUMAN-ANNOTATED relation from 9 relation types
(+ Other), with direction, e.g. Cause-Effect(e1,e2).

Label mapping used in this project (documented, deterministic):
    Cause-Effect(e1,e2)  -> CAUSE_E1_E2   (relation_class=causal, direction=entity_1->entity_2)
    Cause-Effect(e2,e1)  -> CAUSE_E2_E1   (relation_class=causal, direction=entity_2->entity_1)
    every other relation (Other, Component-Whole, Entity-Origin, ...) -> NONE (no causal relation)

SemEval has NO 'association only' category, so ASSOCIATION_ONLY is never learned from it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["NONE", "CAUSE_E1_E2", "CAUSE_E2_E1"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

BASE = "https://raw.githubusercontent.com/mehdi1514/SemEval-Relation-Extraction/main"
URLS = {
    "train": f"{BASE}/SemEval2010_task8_training/TRAIN_FILE.TXT",
    "test": f"{BASE}/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
}
_ENT = re.compile(r"</?e[12]>")


def download(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in URLS.items():
        p = raw_dir / f"semeval2010_{name}.txt"
        if not p.exists():
            urllib.request.urlretrieve(url, p)  # noqa: S310 (fixed https URL)
        out[name] = p
    return out


def map_label(raw: str) -> str:
    """Map an original SemEval label string to the project's 3-way directional label."""
    raw = raw.strip()
    if raw == "Cause-Effect(e1,e2)":
        return "CAUSE_E1_E2"
    if raw == "Cause-Effect(e2,e1)":
        return "CAUSE_E2_E1"
    return "NONE"


def to_marked(sentence: str) -> str:
    """<e1>x</e1> -> [E1] x [/E1] ; <e2>y</e2> -> [E2] y [/E2]."""
    s = sentence.replace("<e1>", "[E1] ").replace("</e1>", " [/E1]")
    s = s.replace("<e2>", "[E2] ").replace("</e2>", " [/E2]")
    return " ".join(s.split())


def parse_file(path: Path) -> list[dict]:
    lines = [l.rstrip("\r\n") for l in path.read_text(encoding="utf-8").splitlines()]
    rows, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        idx, sent = lines[i].split("\t", 1)
        sent = sent.strip().strip('"')
        raw = lines[i + 1].strip()
        rows.append({
            "id": int(idx), "text": to_marked(sent), "plain_text": _ENT.sub("", sent),
            "original_label": raw, "label": map_label(raw),
        })
        i += 4
    return rows


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", _ENT.sub("", t).lower()).strip()


def leakage_safe_split(train_rows: list[dict], dev_fraction: float = 0.1, seed: int = 42):
    """
    Stratified-by-label split in which identical (normalised) sentences always stay together,
    so the same sentence can never appear in both train and dev. SemEval has no document ids;
    sentence identity is the finest leakage unit available.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        groups[_norm(r["plain_text"])].append(r)
    by_label: dict[str, list[str]] = defaultdict(list)
    for key, rs in groups.items():
        by_label[Counter(r["label"] for r in rs).most_common(1)[0][0]].append(key)
    rng = random.Random(seed)
    dev_keys = set()
    for lab, keys in by_label.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        dev_keys.update(keys[: int(round(len(keys) * dev_fraction))])
    tr, dv = [], []
    for key, rs in groups.items():
        (dv if key in dev_keys else tr).extend(rs)
    return sorted(tr, key=lambda r: r["id"]), sorted(dv, key=lambda r: r["id"])


def quality_report(splits: dict[str, list[dict]]) -> dict:
    rep = {"label_distribution": {}, "empty_text": 0, "missing_markers": 0, "overlap_train_dev": 0,
           "overlap_train_test": 0}
    for name, rows in splits.items():
        rep["label_distribution"][name] = dict(Counter(r["label"] for r in rows))
        rep.setdefault("sizes", {})[name] = len(rows)
        for r in rows:
            rep["empty_text"] += int(not r["text"].strip())
            rep["missing_markers"] += int(not all(m in r["text"] for m in ("[E1]", "[/E1]", "[E2]", "[/E2]")))
    n = lambda rows: {_norm(r["plain_text"]) for r in rows}  # noqa: E731
    rep["overlap_train_dev"] = len(n(splits["train"]) & n(splits["dev"]))
    rep["overlap_train_test"] = len(n(splits["train"]) & n(splits["test"]))
    return rep


def prepare(raw_dir: Path, out_dir: Path, dev_fraction: float = 0.1, seed: int = 42) -> dict:
    paths = download(raw_dir)
    train_all, test = parse_file(paths["train"]), parse_file(paths["test"])
    train, dev = leakage_safe_split(train_all, dev_fraction, seed)
    # Official test sentences that also occur in train are dropped from test (strict leakage guard).
    train_keys = {_norm(r["plain_text"]) for r in train + dev}
    test_clean = [r for r in test if _norm(r["plain_text"]) not in train_keys]
    splits = {"train": train, "dev": dev, "test": test_clean}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"semeval_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = quality_report(splits)
    rep.update({"seed": seed, "dev_fraction": dev_fraction, "test_dropped_for_leakage": len(test) - len(test_clean),
                "source": "SemEval-2010 Task 8 (human-annotated gold)", "label_mapping": {
                    "Cause-Effect(e1,e2)": "CAUSE_E1_E2", "Cause-Effect(e2,e1)": "CAUSE_E2_E1", "all others": "NONE"}})
    (out_dir / "semeval_quality_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def load_split(out_dir: Path, name: str) -> list[dict]:
    with open(out_dir / f"semeval_{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(prepare(root / "data/raw", root / "data/processed"), indent=2))
