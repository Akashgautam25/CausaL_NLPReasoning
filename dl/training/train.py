"""
Fine-tune DistilBERT for directional causal relation extraction on SemEval-2010 Task 8.

Mac (M3 Pro, MPS) -- the real experiment, with pretrained weights:
    python -m dl.training.train --model distilbert-base-uncased --epochs 4 --batch-size 16

Offline pipeline check (no pretrained weights; random init, NOT transfer learning):
    python -m dl.training.train --offline-smoke --epochs 3
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.extractors.distilbert.dataset import RelationDataset, make_collate, marker_ids
from src.extractors.distilbert.model import (build_offline_smoke, build_pretrained, pick_device, save_checkpoint)
from src.preprocessing.semeval import LABELS, LABEL2ID, load_split, prepare
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    model.eval()
    ys, ps, tot, n = [], [], 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(**b, class_weights=class_weights)
        tot += float(out["loss"]) * len(b["labels"])
        n += len(b["labels"])
        ps += out["logits"].argmax(-1).cpu().tolist()
        ys += b["labels"].cpu().tolist()
    return tot / max(n, 1), float(f1_score(ys, ps, average="macro", labels=list(range(len(LABELS))), zero_division=0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--offline-smoke", action="store_true", help="random-init small model; pipeline check only")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-train", type=int, default=None, help="subsample train (stratified-agnostic, seeded)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    set_seed(a.seed)
    device = pick_device(a.device)
    if device.type == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    data_dir = ROOT / "data/processed"
    if not (data_dir / "semeval_train.jsonl").exists():
        prepare(ROOT / "data/raw", data_dir, seed=a.seed)
    train_rows, dev_rows = load_split(data_dir, "train"), load_split(data_dir, "dev")
    if a.max_train and a.max_train < len(train_rows):
        import random
        train_rows = random.Random(a.seed).sample(train_rows, a.max_train)

    tag = "smoke_random_init" if a.offline_smoke else "distilbert_base_uncased"
    out_dir = Path(a.out) if a.out else ROOT / "models" / tag
    if a.offline_smoke:
        tok, model = build_offline_smoke([r["text"] for r in train_rows], len(LABELS), out_dir / "_tok", seed=a.seed)
    else:
        tok, model = build_pretrained(a.model, len(LABELS), a.local_files_only)
    model.to(device)

    m = marker_ids(tok)
    coll = make_collate(tok.pad_token_id, m["[E1]"], m["[E2]"])
    tr_ds = RelationDataset(train_rows, tok, a.max_length)
    dv_ds = RelationDataset(dev_rows, tok, a.max_length)
    g = torch.Generator().manual_seed(a.seed)
    tr_dl = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, collate_fn=coll, generator=g)
    dv_dl = DataLoader(dv_ds, batch_size=64, shuffle=False, collate_fn=coll)

    # Inverse-frequency class weights (normalised to mean 1) to counter the ~87% NONE imbalance.
    counts = np.bincount(tr_ds.labels, minlength=len(LABELS)).astype(float)
    cw = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float, device=device)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": a.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=a.lr)
    total = len(tr_dl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * total), total)

    hist, best, bad, t_start = [], -1.0, 0, time.perf_counter()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run, n = time.perf_counter(), 0.0, 0
        for b in tr_dl:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b, class_weights=cw)["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(b["labels"])
            n += len(b["labels"])
        vl, vf1 = evaluate(model, dv_dl, device, cw)
        rec = {"epoch": ep, "train_loss": run / n, "val_loss": vl, "val_macro_f1": vf1,
               "epoch_seconds": time.perf_counter() - t0}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if vf1 > best:
            best, bad = vf1, 0
            save_checkpoint(model, tok, out_dir, {
                "num_labels": len(LABELS), "labels": LABELS, "max_length": a.max_length,
                "base_model": "random-init-small-DistilBERT-architecture" if a.offline_smoke else a.model,
                "pretrained": not a.offline_smoke, "best_epoch": ep, "best_val_macro_f1": vf1, "seed": a.seed})
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break
    summary = {"args": vars(a), "device": str(device), "platform": platform.platform(),
               "torch": torch.__version__, "pretrained": not a.offline_smoke,
               "n_train": len(train_rows), "n_dev": len(dev_rows), "class_weights": cw.cpu().tolist(),
               "total_training_seconds": time.perf_counter() - t_start, "history": hist,
               "best_val_macro_f1": best,
               "WARNING": "random-init smoke run: pipeline verification only, NOT transfer learning"
               if a.offline_smoke else None}
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    res = ROOT / "dl/results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"training_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("saved", out_dir)


if __name__ == "__main__":
    main()
