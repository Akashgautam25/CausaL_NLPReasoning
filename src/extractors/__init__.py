"""
Relation extraction architectures (Rule-based, DistilBERT, and LLM).

Submodules:
    - base: BaseExtractor abstract class defining the unified prediction contract.
    - rules_extractor: Dependency syntax and trigger-phrase matcher.
    - distilbert_extractor: Fine-tuned biomedical transformer classifier.
    - llm_extractor: Few-shot prompting with structured epistemic reasoning.
"""

from src.extractors.base import BaseExtractor

__all__ = [
    "BaseExtractor",
]
