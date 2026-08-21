# Stock research agent policy

You are operating an isolated offline experiment branch. Read `research/README.md`
and the immutable evaluator before changing anything.

Rules:

1. Modify only the registered candidate module for an experiment.
2. Never modify snapshots, fold definitions, target construction, baselines,
   metric formulas, promotion thresholds, production code, or deployment files.
3. State one hypothesis before each experiment.
4. Run the correctness smoke level before screening or confirmation.
5. Record every completed, failed, cancelled, timed-out, and out-of-memory run
   in the append-only ledger.
6. Keep an improvement only when its required fidelity gates pass; otherwise
   leave the isolated branch at its prior candidate.
7. Do not inspect or tune against the locked test period.
8. Do not push, deploy, or label a candidate as production-ready. Nominate it
   with its snapshot, folds, seeds, metrics, resource use, and failure history.

Persistence and Ridge are mandatory references. A neural candidate must beat
the relevant baseline on both relative MAE and relative RMSE before receiving
confirmation budget.

9. Multiplicity policy (screen-then-single-holdout-v1): fold-level gates are
   exploratory screening across many configurations; each surviving
   (family, ticker, horizon) claim requires exactly ONE locked multi-window
   holdout on a frozen snapshot. Re-running the holdout after observing
   results invalidates prior certification for that claim. Production-facing
   claims require re-certification on new untouched data. Records created
   before provenance tracking (schema v1, commit/snapshot "unknown") are
   labelled LEGACY_UNAUDITED in generated reports and must not be presented
   as certified evidence.
