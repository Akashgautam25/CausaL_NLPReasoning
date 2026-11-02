"""
Scientific literature retrieval and frozen corpus management via Europe PMC.

Submodules:
    - epmc_client: REST client for Europe PMC search and abstract retrieval.
    - corpus: CorpusManager handling live queries and offline frozen datasets.
    - manifest: Corpus manifest serialization and exclusion tracking.
    - dedup: Document deduplication across queries and identifiers.
    - licence: Open-access licence validation.
"""

from src.retrieval.corpus import CorpusManager, NetworkAccessForbiddenError

__all__ = [
    "CorpusManager",
    "NetworkAccessForbiddenError",
]
