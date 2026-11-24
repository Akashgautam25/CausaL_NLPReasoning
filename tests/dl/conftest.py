"""Shared fixtures. Tests never need network or pretrained weights: they build a tiny random-init model."""
import pytest
import torch

from src.extractors.distilbert.model import build_offline_smoke, save_checkpoint
from src.preprocessing.semeval import LABELS

TEXTS = [
    "The [E1] smoking [/E1] causes [E2] cancer [/E2] in many patients .",
    "The [E1] burst [/E1] has been caused by water hammer [E2] pressure [/E2] .",
    "A [E1] cat [/E1] sat on the [E2] mat [/E2] near the door .",
] * 40


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    tok, model = build_offline_smoke(TEXTS, len(LABELS), d / "tok", vocab_size=300)
    save_checkpoint(model, tok, d / "m", {"num_labels": len(LABELS), "labels": LABELS, "max_length": 64,
                                           "pretrained": False})
    return d / "m"
