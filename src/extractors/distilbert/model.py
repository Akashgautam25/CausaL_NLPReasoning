"""DistilBERT + entity-aware classification head, and tokenizer/model loading utilities."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel

from src.extractors.distilbert.markers import MARKER_TOKENS


def pick_device(prefer: str | None = None) -> torch.device:
    """Apple MPS if available, else CUDA, else CPU (override with prefer='cpu'|'mps'|'cuda')."""
    if prefer:
        return torch.device(prefer)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DistilBertRelationClassifier(nn.Module):
    """
    Representation = concat( h[CLS], h[[E1]], h[[E2]] ) from DistilBERT's last layer -> dropout -> linear.
    Pooling the hidden states at the entity-start markers (as in R-BERT / "Matching the Blanks"
    entity-start pooling) lets the head see *which* entity is which, which is what direction needs.
    """

    def __init__(self, encoder: DistilBertModel, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.dim
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(3 * h, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos, labels=None, class_weights=None):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ar = torch.arange(hs.size(0), device=hs.device)
        feats = torch.cat([hs[:, 0], hs[ar, e1_pos], hs[ar, e2_pos]], dim=-1)
        logits = self.classifier(self.drop(feats))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
        return {"loss": loss, "logits": logits}


def build_pretrained(model_name: str, num_labels: int, local_files_only: bool = False):
    """Real transfer learning: load pretrained `distilbert-base-uncased` weights + tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    enc = DistilBertModel.from_pretrained(model_name, local_files_only=local_files_only)
    enc.resize_token_embeddings(len(tok))
    return tok, DistilBertRelationClassifier(enc, num_labels)


def build_offline_smoke(texts: list[str], num_labels: int, workdir: Path, vocab_size: int = 8000, seed: int = 42):
    """
    PIPELINE-VERIFICATION ONLY. Used when pretrained weights cannot be downloaded (e.g. no access to
    huggingface.co). Trains a WordPiece vocabulary on the local training text and builds a small,
    RANDOMLY INITIALISED model with the DistilBERT architecture. This is NOT transfer learning
    and its scores must never be reported as the performance of pretrained DistilBERT.
    """
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertTokenizerFast

    workdir.mkdir(parents=True, exist_ok=True)
    wp = BertWordPieceTokenizer(lowercase=True)
    plain = [t.replace("[E1]", " ").replace("[/E1]", " ").replace("[E2]", " ").replace("[/E2]", " ") for t in texts]
    wp.train_from_iterator(plain, vocab_size=vocab_size, min_frequency=2)
    wp.save_model(str(workdir))
    tok = BertTokenizerFast(vocab_file=str(workdir / "vocab.txt"), do_lower_case=True)
    tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    torch.manual_seed(seed)
    cfg = DistilBertConfig(vocab_size=len(tok), dim=128, hidden_dim=512, n_layers=2, n_heads=4,
                           max_position_embeddings=256, pad_token_id=tok.pad_token_id)
    return tok, DistilBertRelationClassifier(DistilBertModel(cfg), num_labels)


def save_checkpoint(model, tokenizer, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    model.encoder.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_checkpoint(ckpt_dir: Path, device: torch.device | None = None):
    """Rebuild architecture from the saved config (no network needed) and load fine-tuned weights."""
    ckpt_dir = Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    cfg = DistilBertConfig.from_pretrained(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    model = DistilBertRelationClassifier(DistilBertModel(cfg), meta["num_labels"])
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.to(device or pick_device()).eval()
    return tok, model, meta
