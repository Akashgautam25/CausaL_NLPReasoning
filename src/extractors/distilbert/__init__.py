"""Genuine fine-tunable DistilBERT relation extractor (entity-marker + entity-pooled head)."""
from src.extractors.distilbert.markers import MARKER_TOKENS, insert_markers_by_offsets
from src.extractors.distilbert.model import DistilBertRelationClassifier, pick_device

__all__ = ["MARKER_TOKENS", "insert_markers_by_offsets", "DistilBertRelationClassifier", "pick_device"]
# FineTunedDistilBERTExtractor lives in src.extractors.distilbert.extractor (imported lazily: it needs torch).
