"""
Annotation tools for biomedical relation extraction and evaluation.

Submodules:
    - exporter: Export candidate extraction pairs to CSV for manual annotation.
    - importer: Import completed manual annotations with schema validation.
    - kappa: Compute Cohen's / Fleiss' Kappa inter-annotator agreement metrics.
    - sampler: Stratified document and candidate pair partitioning.
    - prelabel: Heuristic pre-labeling utilities.
"""

from src.annotation.exporter import export_candidates_to_csv
from src.annotation.importer import import_annotations_from_csv
from src.annotation.kappa import compute_agreement, save_agreement_report
from src.annotation.sampler import partition_documents

__all__ = [
    "export_candidates_to_csv",
    "import_annotations_from_csv",
    "compute_agreement",
    "save_agreement_report",
    "partition_documents",
]
