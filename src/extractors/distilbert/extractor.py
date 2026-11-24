"""BaseExtractor implementation backed by a fine-tuned DistilBERT checkpoint."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from src.extractors.base import BaseExtractor
from src.extractors.distilbert.dataset import RelationDataset, make_collate, marker_ids
from src.extractors.distilbert.markers import insert_markers_by_offsets
from src.extractors.distilbert.model import load_checkpoint, pick_device
from src.preprocessing.semeval import ID2LABEL
from src.schemas.prediction import (Direction, LanguageStatus, Polarity, Prediction, Provenance, RelationClass,
                                    RelationLabel, ReviewStatus)
from src.schemas.text import CandidatePair


class FineTunedDistilBERTExtractor(BaseExtractor):
    """
    Genuine neural extractor. Output classes: NONE / CAUSE_E1_E2 / CAUSE_E2_E1 -> mapped onto the
    project schema as relation_class in {none, causal} + direction. It never emits ASSOCIATION_ONLY
    (no gold association labels exist in the training data). Polarity and hedging are NOT modelled:
    polarity='mixed_or_unclear', language_status='affirmed' are explicit defaults, flagged in raw_output.
    """

    name = "FineTuned-DistilBERT"
    version = "2.0.0"

    def __init__(self, checkpoint_dir: str | Path, device: str | None = None, batch_size: int = 32):
        self.device = pick_device(device)
        self.tokenizer, self.model, self.meta = load_checkpoint(Path(checkpoint_dir), self.device)
        self.max_length = int(self.meta.get("max_length", 128))
        self.batch_size = batch_size
        self.pretrained = bool(self.meta.get("pretrained", False))
        self.last_inference_seconds = 0.0

    @torch.no_grad()
    def predict_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, len(ID2LABEL))
        rows = [{"text": t, "label": "NONE"} for t in texts]
        ds = RelationDataset(rows, self.tokenizer, self.max_length, with_labels=False)
        m = marker_ids(self.tokenizer)
        coll = make_collate(self.tokenizer.pad_token_id, m["[E1]"], m["[E2]"])
        probs = []
        for i in range(0, len(ds), self.batch_size):
            b = coll([ds[j] for j in range(i, min(i + self.batch_size, len(ds)))])
            b = {k: v.to(self.device) for k, v in b.items()}
            probs.append(torch.softmax(self.model(**b)["logits"], -1).cpu())
        return torch.cat(probs) if probs else torch.zeros(0, len(ID2LABEL))

    def _to_prediction(self, pair: CandidatePair, probs: torch.Tensor | None, marked: str | None, err: str | None):
        prov = Provenance(doc_id=pair.doc_id, sent_id=pair.sent_id,
                          url=f"https://europepmc.org/article/MED/{pair.doc_id.split(':')[-1]}",
                          pmid=pair.doc_id.split(":")[-1] if pair.doc_id.startswith("PMID:") else None,
                          sentence_text=pair.sentence_text,
                          char_spans=[pair.entity_1.start_char, pair.entity_2.end_char],
                          extractor_name=self.name, confidence=0.0, review_status=ReviewStatus.UNREVIEWED)
        if probs is None:
            label = RelationLabel(relation_class=RelationClass.NONE, direction=Direction.NOT_APPLICABLE,
                                  polarity=Polarity.NOT_APPLICABLE, language_status=LanguageStatus.AFFIRMED,
                                  confidence=0.0, reasoning=f"unscorable: {err}", provenance=prov)
            return Prediction(pair_id=pair.pair_id, extractor_name=self.name, extractor_version=self.version,
                              label=label, raw_output={"error": err}, validation_status="schema_error",
                              error_message=err)
        k = int(probs.argmax())
        name = ID2LABEL[k]
        conf = float(probs[k])
        prov.confidence = conf
        if name == "NONE":
            rc, d, pol = RelationClass.NONE, Direction.NOT_APPLICABLE, Polarity.NOT_APPLICABLE
        else:
            rc, pol = RelationClass.CAUSAL, Polarity.MIXED_OR_UNCLEAR
            d = Direction.E1_TO_E2 if name == "CAUSE_E1_E2" else Direction.E2_TO_E1
        label = RelationLabel(relation_class=rc, direction=d, polarity=pol, language_status=LanguageStatus.AFFIRMED,
                              confidence=conf, reasoning=f"DistilBERT argmax={name} p={conf:.3f}", provenance=prov)
        return Prediction(pair_id=pair.pair_id, extractor_name=self.name, extractor_version=self.version, label=label,
                          raw_output={"marked_text": marked, "probs": {ID2LABEL[i]: round(float(p), 5) for i, p in enumerate(probs)},
                                      "polarity_and_language_status": "defaults_not_modelled",
                                      "pretrained_weights": self.pretrained})

    def predict(self, pairs: list[CandidatePair]) -> list[Prediction]:
        marked, bad = [], {}
        for i, p in enumerate(pairs):
            try:
                marked.append(insert_markers_by_offsets(p.sentence_text, p.entity_1.start_char, p.entity_1.end_char,
                                                        p.entity_2.start_char, p.entity_2.end_char))
            except ValueError as e:
                marked.append(None)
                bad[i] = str(e)
        ok = [i for i, m in enumerate(marked) if m is not None]
        t0 = time.perf_counter()
        probs = self.predict_texts([marked[i] for i in ok])
        self.last_inference_seconds = time.perf_counter() - t0
        pm = {i: probs[j] for j, i in enumerate(ok)}
        return [self._to_prediction(p, pm.get(i), marked[i], bad.get(i)) for i, p in enumerate(pairs)]
