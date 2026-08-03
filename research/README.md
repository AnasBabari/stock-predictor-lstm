# Stock autoresearch harness

This is an isolated offline research harness inspired by the MIT-licensed `autoresearch-win-rtx` experiment loop. It does not run in Render, the browser, or the production request path.

The evaluator uses frozen snapshots, cumulative log-return targets, expanding time-ordered folds, horizon-dependent purge boundaries, persistence-relative MAE/RMSE, and append-only experiment records. The initial runner supports persistence and Ridge so the evaluator can be verified before neural candidates are introduced.

The local RTX 2060 has 6 GB VRAM, below the external fork's supported Turing floor. Future PyTorch candidates must use the stock harness's conservative runtime profile and official CUDA wheels.

Run from the repository root with `PYTHONPATH=research` after preparing a frozen CSV snapshot. Do not commit generated snapshots, model outputs, credentials, or browser weights.
