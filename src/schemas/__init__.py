"""
Core data contracts and Pydantic schemas for documents, mentions, relations, and predictions.

Submodules:
    - document: Document representation and corpus manifests.
    - text: Sentences, entity mentions, and candidate relation pairs.
    - prediction: Relation labels, polarities, language status, and extractor predictions.
"""

from src.schemas.document import Document
from src.schemas.prediction import (
    Direction,
    LanguageStatus,
    Polarity,
    Prediction,
    Provenance,
    RelationClass,
    RelationLabel,
    ReviewStatus,
)
from src.schemas.text import CandidatePair, Mention, Sentence

__all__ = [
    "Document",
    "Sentence",
    "Mention",
    "CandidatePair",
    "RelationClass",
    "Direction",
    "Polarity",
    "LanguageStatus",
    "ReviewStatus",
    "Provenance",
    "RelationLabel",
    "Prediction",
]
