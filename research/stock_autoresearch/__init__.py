"""Stock Autoresearch Package."""

from __future__ import annotations

from .candidates import (
    LEGACY_FAMILY_ALIASES,
    Candidate,
    CompactMLPCandidate,
    DLinearCandidate,
    ElasticNetCandidate,
    PersistenceCandidate,
    RandomFeaturesRidgeCandidate,
    RidgeCandidate,
    canonical_family,
)
from .config import EVALUATION_POLICY, RUNTIME_BUDGET, EvaluationPolicy, RuntimeBudget
from .controller import ExperimentController, SubprocessResult, run_isolated_candidate_eval
from .data import Snapshot, build_examples, expanding_folds
from .evaluation import EvaluationResult, FoldResult, evaluate_candidate
from .ledger import append_record, export_tsv_summary, generate_markdown_report
from .metrics import classification_metrics, regression_metrics
from .parity import ParityResult, make_parity_fixture, verify_prediction_parity
from .resources import ResourceSample, reset_cuda_memory, sample_cuda_memory

__all__ = [
    "Candidate",
    "PersistenceCandidate",
    "RidgeCandidate",
    "ElasticNetCandidate",
    "CompactMLPCandidate",
    "DLinearCandidate",
    "RandomFeaturesRidgeCandidate",
    "LEGACY_FAMILY_ALIASES",
    "canonical_family",
    "RuntimeBudget",
    "EvaluationPolicy",
    "RUNTIME_BUDGET",
    "EVALUATION_POLICY",
    "Snapshot",
    "build_examples",
    "expanding_folds",
    "evaluate_candidate",
    "FoldResult",
    "EvaluationResult",
    "regression_metrics",
    "classification_metrics",
    "append_record",
    "export_tsv_summary",
    "generate_markdown_report",
    "ResourceSample",
    "sample_cuda_memory",
    "reset_cuda_memory",
    "ExperimentController",
    "SubprocessResult",
    "run_isolated_candidate_eval",
    "ParityResult",
    "make_parity_fixture",
    "verify_prediction_parity",
]
