"""Torch dataset: tokenisation with entity markers, attention masks, label encoding."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.extractors.distilbert.markers import MARKER_TOKENS
from src.preprocessing.semeval import LABEL2ID


def marker_ids(tokenizer) -> dict[str, int]:
    return {t: tokenizer.convert_tokens_to_ids(t) for t in MARKER_TOKENS}


def encode_texts(tokenizer, texts: list[str], max_length: int):
    return tokenizer(texts, truncation=True, max_length=max_length, padding=False)


class RelationDataset(Dataset):
    """Holds pre-tokenised, marker-annotated sentences. Padding is done per batch by `collate`."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 128, with_labels: bool = True):
        self.enc = encode_texts(tokenizer, [r["text"] for r in rows], max_length)
        self.labels = [LABEL2ID[r["label"]] for r in rows] if with_labels else None
        self.pad_id = tokenizer.pad_token_id
        m = marker_ids(tokenizer)
        self.e1_id, self.e2_id = m["[E1]"], m["[E2]"]

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {"input_ids": self.enc["input_ids"][i]}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def make_collate(pad_id: int, e1_id: int, e2_id: int):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        p1 = torch.zeros(len(batch), dtype=torch.long)  # position of [E1]; 0 (=[CLS]) if truncated away
        p2 = torch.zeros(len(batch), dtype=torch.long)
        for i, b in enumerate(batch):
            x = torch.tensor(b["input_ids"])
            ids[i, : len(x)] = x
            mask[i, : len(x)] = 1
            w1, w2 = (x == e1_id).nonzero(), (x == e2_id).nonzero()
            if len(w1):
                p1[i] = w1[0, 0]
            if len(w2):
                p2[i] = w2[0, 0]
        out = {"input_ids": ids, "attention_mask": mask, "e1_pos": p1, "e2_pos": p2}
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return out
    return collate
