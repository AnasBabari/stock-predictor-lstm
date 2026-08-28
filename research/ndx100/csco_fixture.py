"""Immutable CSCO golden fixture — deterministic offline regression case.

This module pins network-independent inputs and expected forecasts, calendars
and metrics for the CSCO 2026-07-20..2026-07-24 target week.

Design rules enforced here:

* **One canonical format.** The committed CSV is the single source of truth.
  No secondary serialized copy exists, so the two cannot drift apart.
* **Hash before parse.** The canonical SHA-256 is verified against the manifest
  *before* any deserialization. A corrupt or substituted file fails closed.
* **No silent fallback.** Loading never falls back to the network. Live data
  is reachable only through the explicit ``--live`` CLI flag.
* **Evidence role.** This is a software-regression golden case, not
  certification-grade market evidence. See ``csco_golden_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

TICKER = "CSCO"
TRAIN_END = pd.Timestamp("2026-07-17")
TARGET_DAYS = tuple(
    pd.Timestamp(day)
    for day in (
        "2026-07-20",
        "2026-07-21",
        "2026-07-22",
        "2026-07-23",
        "2026-07-24",
    )
)
HORIZON = len(TARGET_DAYS)
SEED = 41
LAGS = (1, 2, 3, 5, 10)
WINDOW = 20
CPU_THREADS = 1

FEATURES = tuple(
    [f"lag_{lag}" for lag in LAGS] + ["vol_5", "vol_20", "ma5_ratio", "ma20_ratio", "volume_z"]
)

FIXTURE_DIR = Path("data/fixtures")
CANONICAL_CSV_NAME = "csco_golden_2015_2026.csv"
MANIFEST_NAME = "csco_golden_manifest.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CSCO_FIXTURE_CSV = FIXTURE_DIR / CANONICAL_CSV_NAME
CSCO_FIXTURE_MANIFEST = FIXTURE_DIR / MANIFEST_NAME

EXPECTED_ROW_COUNT = 2906
EXPECTED_FIRST_SESSION = pd.Timestamp("2015-01-02")
EXPECTED_LAST_SESSION = pd.Timestamp("2026-07-24")
EXPECTED_COLUMNS = ("close", "volume")
EXPECTED_CANONICAL_SHA256 = "afc432dead9a7352e78d27e9e031685f0dae994de1d8ac461037bead8d19cf25"
EXPECTED_EVIDENCE_ROLE = "synthetic_software_regression_fixture"

EXPECTED_ACTUAL_CLOSES = (43.30, 44.42, 44.62, 45.36, 45.99)
LAST_TRAIN_CLOSE = 42.57

EXPECTED_RANKING = (
    "lstm_window",
    "drift_random_walk",
    "naive_flat",
    "ridge_lagged_returns",
    "hgb_boosting",
    "seasonal_naive_5d",
)

# Models whose outputs should remain tightly reproducible in the supported
# dependency range. They are not described as bit-for-bit portable because
# BLAS and tree implementations may legitimately differ across platforms.
DETERMINISTIC_MODELS = (
    "naive_flat",
    "drift_random_walk",
    "seasonal_naive_5d",
    "ridge_lagged_returns",
    "hgb_boosting",
)
# Neural outputs are implementation-sensitive; validated loosely, never exactly.
IMPLEMENTATION_SENSITIVE_MODELS = ("lstm_window",)

GOLDEN_PREDICTIONS: dict[str, tuple[float, ...]] = {
    "naive_flat": (42.57, 42.57, 42.57, 42.57, 42.57),
    "drift_random_walk": (42.58, 42.58, 42.59, 42.60, 42.60),
    "seasonal_naive_5d": (42.29, 41.29, 41.61, 41.59, 42.57),
    "ridge_lagged_returns": (42.54, 42.55, 42.56, 42.57, 42.53),
    "hgb_boosting": (42.53, 42.50, 42.55, 42.31, 42.43),
    # Reference environment only: PyTorch 2.13.0+cpu, 1 CPU thread, seed 41.
    "lstm_window": (43.34, 43.74, 44.09, 44.05, 44.04),
}

GOLDEN_METRICS: dict[str, dict[str, float]] = {
    "lstm_window": {"mape_pct": 1.9836, "rmse": 1.1175, "mae": 0.9001, "r2": -0.5092},
    "drift_random_walk": {"mape_pct": 4.7638, "rmse": 2.3299, "mae": 2.1488, "r2": -5.5605},
    "naive_flat": {"mape_pct": 4.8064, "rmse": 2.3511, "mae": 2.1680, "r2": -5.6803},
    "ridge_lagged_returns": {"mape_pct": 4.8515, "rmse": 2.3690, "mae": 2.1881, "r2": -5.7824},
    "hgb_boosting": {"mape_pct": 5.0395, "rmse": 2.4707, "mae": 2.2733, "r2": -6.3773},
    "seasonal_naive_5d": {"mape_pct": 6.3745, "rmse": 3.0261, "mae": 2.8680, "r2": -10.0668},
}

# Tolerances are deliberately split by reproducibility class. Deterministic
# models get tight tolerances; the LSTM gets a tolerance wide enough to survive
# legitimate cross-platform kernel differences without accepting real breakage.
TIGHT_TOLERANCE = {"rtol": 1e-5, "atol": 0.02}
LOOSE_TOLERANCE = {"rtol": 5e-2, "atol": 0.75}
METRIC_TOLERANCE_DETERMINISTIC = 5e-4
METRIC_TOLERANCE_SENSITIVE = 0.5


class FixtureIntegrityError(RuntimeError):
    """Raised when the golden fixture or its manifest fails verification."""


def environment_versions() -> dict[str, str]:
    """Record the versions that produced the golden values."""
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def load_manifest(base_path: Path | None = None) -> dict[str, Any]:
    """Load the committed fixture manifest. Fails closed if absent."""
    root = Path(base_path) if base_path is not None else REPOSITORY_ROOT
    path = root / CSCO_FIXTURE_MANIFEST
    if not path.is_file():
        raise FixtureIntegrityError(f"CSCO fixture manifest not found at {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureIntegrityError(f"CSCO fixture manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FixtureIntegrityError("CSCO fixture manifest must be a JSON object")

    required = {
        "manifest_version",
        "fixture_id",
        "evidence_role",
        "certification_eligible",
        "canonical_format",
        "canonical_file",
        "canonical_sha256",
        "canonical_size_bytes",
        "row_count",
        "date_range",
        "columns",
        "provenance",
        "limitations",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise FixtureIntegrityError(f"CSCO fixture manifest missing fields: {missing}")
    if manifest["evidence_role"] != EXPECTED_EVIDENCE_ROLE:
        raise FixtureIntegrityError(
            f"CSCO fixture evidence role must be {EXPECTED_EVIDENCE_ROLE!r}"
        )
    if manifest["certification_eligible"] is not False:
        raise FixtureIntegrityError(
            "CSCO software-regression fixture cannot be certification eligible"
        )
    if manifest["canonical_format"] != "csv" or manifest["canonical_file"] != CANONICAL_CSV_NAME:
        raise FixtureIntegrityError("CSCO manifest does not identify the canonical CSV fixture")
    if manifest["manifest_version"] != "1.0" or manifest["fixture_id"] != "csco_golden_2015_2026":
        raise FixtureIntegrityError("CSCO manifest version or fixture identity is incompatible")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["canonical_sha256"])):
        raise FixtureIntegrityError("CSCO manifest canonical_sha256 must be lowercase SHA-256 hex")
    if manifest["canonical_sha256"] != EXPECTED_CANONICAL_SHA256:
        raise FixtureIntegrityError(
            "CSCO manifest hash does not match the hash pinned by the loader contract"
        )
    if (
        int(manifest["canonical_size_bytes"]) <= 0
        or int(manifest["row_count"]) != EXPECTED_ROW_COUNT
    ):
        raise FixtureIntegrityError("CSCO manifest size or row-count contract is incompatible")
    if manifest["date_range"] != {
        "first_session": EXPECTED_FIRST_SESSION.date().isoformat(),
        "last_session": EXPECTED_LAST_SESSION.date().isoformat(),
    }:
        raise FixtureIntegrityError("CSCO manifest date range is incompatible")
    if tuple(manifest["columns"].keys()) != EXPECTED_COLUMNS:
        raise FixtureIntegrityError("CSCO manifest column ordering is incompatible")
    if not isinstance(manifest["provenance"], dict) or not manifest["provenance"]:
        raise FixtureIntegrityError("CSCO manifest provenance must be a non-empty object")
    if not isinstance(manifest["limitations"], list) or not manifest["limitations"]:
        raise FixtureIntegrityError("CSCO manifest limitations must be a non-empty list")
    return manifest


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_canonical_fixture(
    csv_path: Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the canonical CSV against the manifest *before* parsing it.

    Returns the manifest on success. Raises :class:`FixtureIntegrityError` on
    any mismatch, because a fixture whose identity cannot be proven must never
    be allowed to drive a benchmark result.
    """
    manifest = manifest if manifest is not None else load_manifest(csv_path.parents[2])

    if not csv_path.is_file():
        raise FixtureIntegrityError(f"CSCO canonical fixture not found at {csv_path}")

    expected_sha = str(manifest["canonical_sha256"])
    actual_sha = sha256_file(csv_path)
    if actual_sha != expected_sha:
        raise FixtureIntegrityError(
            f"CSCO fixture hash mismatch: manifest declares {expected_sha}, "
            f"file at {csv_path} hashes to {actual_sha}. The fixture has been "
            f"modified, truncated or corrupted."
        )

    expected_size = manifest.get("canonical_size_bytes")
    if expected_size is not None and csv_path.stat().st_size != int(expected_size):
        raise FixtureIntegrityError(
            f"CSCO fixture size mismatch: expected {expected_size}, got {csv_path.stat().st_size}"
        )
    return manifest


def load_csco_golden_history(base_path: Path | None = None) -> pd.DataFrame:
    """Load pinned offline CSCO history strictly without network access.

    The canonical hash is verified before deserialization. There is no fallback
    of any kind: a missing, corrupt or substituted fixture raises.
    """
    root = Path(base_path) if base_path is not None else REPOSITORY_ROOT
    csv_path = root / CSCO_FIXTURE_CSV
    manifest = verify_canonical_fixture(csv_path, load_manifest(root))

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as exc:
        raise FixtureIntegrityError(
            f"CSCO fixture at {csv_path} passed hash verification but could not be parsed: {exc}"
        ) from exc

    if tuple(df.columns) != EXPECTED_COLUMNS:
        raise FixtureIntegrityError(
            f"CSCO fixture columns mismatch: expected {EXPECTED_COLUMNS}, got {tuple(df.columns)}"
        )
    if df.isna().any().any():
        raise FixtureIntegrityError("CSCO fixture contains missing close or volume values")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise FixtureIntegrityError("CSCO fixture contains non-finite close or volume values")

    df = df.copy()
    try:
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    except (TypeError, ValueError) as exc:
        raise FixtureIntegrityError(
            f"CSCO fixture index is not a valid session date index: {exc}"
        ) from exc
    if not df.index.is_monotonic_increasing:
        raise FixtureIntegrityError("CSCO fixture session dates are not strictly chronological")
    if not df.index.is_unique:
        raise FixtureIntegrityError("CSCO fixture contains duplicate session dates")

    expected_rows = int(manifest.get("row_count", EXPECTED_ROW_COUNT))
    if len(df) != expected_rows:
        raise FixtureIntegrityError(
            f"CSCO fixture row count mismatch: expected {expected_rows}, got {len(df)}"
        )
    if df.index.min() != EXPECTED_FIRST_SESSION or df.index.max() != EXPECTED_LAST_SESSION:
        raise FixtureIntegrityError(
            f"CSCO fixture date range mismatch: expected "
            f"{EXPECTED_FIRST_SESSION.date()}..{EXPECTED_LAST_SESSION.date()}, got "
            f"{df.index.min().date()}..{df.index.max().date()}"
        )
    if not (df["close"] > 0).all():
        raise FixtureIntegrityError("CSCO fixture contains non-positive closes")
    if not (df["volume"] >= 0).all():
        raise FixtureIntegrityError("CSCO fixture contains negative volume")
    return df


def compute_csco_feature_frame(
    closes: np.ndarray,
    volumes: np.ndarray,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Causal per-session features; every row uses only data at or before it."""
    frame = pd.DataFrame(
        {"close": closes, "volume": volumes.astype(float)},
        index=index,
    )
    log_close = np.log(frame["close"])
    log_return = log_close.diff()
    features = pd.DataFrame(index=frame.index)
    for lag in LAGS:
        features[f"lag_{lag}"] = log_return.shift(lag)
    features["vol_5"] = log_return.rolling(5).std()
    features["vol_20"] = log_return.rolling(20).std()
    features["ma5_ratio"] = frame["close"] / frame["close"].rolling(5).mean() - 1.0
    features["ma20_ratio"] = frame["close"] / frame["close"].rolling(20).mean() - 1.0
    volume_mean = frame["volume"].rolling(20).mean()
    features["volume_z"] = frame["volume"] / volume_mean - 1.0
    return features


def _recursive_forecast(
    name: str,
    history: pd.DataFrame,
    predict_step: Callable[[np.ndarray], float],
) -> np.ndarray:
    closes = history["close"].to_numpy(dtype=float).copy()
    volumes = history["volume"].to_numpy(dtype=float).copy()
    predictions: list[float] = []
    for _session in range(HORIZON):
        features = compute_csco_feature_frame(closes, volumes)
        row = features.iloc[-1]
        if row[list(FEATURES)].isna().any():
            raise RuntimeError(f"{name}: feature row is incomplete")
        next_log_return = float(predict_step(row[list(FEATURES)].to_numpy(dtype=float)))
        if not np.isfinite(next_log_return):
            raise RuntimeError(f"{name}: non-finite prediction")
        next_close = float(closes[-1]) * float(np.exp(next_log_return))
        predictions.append(next_close)
        closes = np.append(closes, next_close)
        volumes = np.append(volumes, volumes[-1])
    return np.asarray(predictions, dtype=float)


def _train_rows(history: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = compute_csco_feature_frame(
        history["close"].to_numpy(),
        history["volume"].to_numpy(),
        index=history.index,
    )
    next_return = np.log(history["close"]).diff().shift(-1)
    valid = features[list(FEATURES)].notna().all(axis=1) & next_return.notna()
    valid &= history.index.to_numpy() < np.datetime64(TRAIN_END)
    x = features.loc[valid, list(FEATURES)].to_numpy(dtype=float)
    y = next_return.loc[valid].to_numpy(dtype=float)
    return features, x, y


class ReturnLSTM(nn.Module):
    def __init__(self, feature_count: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_count, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(batch)
        return self.head(output[:, -1, :]).squeeze(-1)


def deterministic_mode(enabled: bool = True, seed: int = SEED) -> None:
    """Constrain PyTorch for reproducible training where the backend allows.

    ``use_deterministic_algorithms`` raises for operations with no
    deterministic implementation, so it is applied only around the LSTM fit and
    never silently downgraded.
    """
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(enabled)
    except (RuntimeError, AttributeError) as exc:  # pragma: no cover - backend dependent
        raise RuntimeError(
            f"Deterministic algorithms unavailable on this backend: {exc}. "
            f"Neural golden values cannot be reproduced here."
        ) from exc


def train_csco_lstm(
    history: pd.DataFrame,
    device: str | torch.device = "cpu",
    seed: int = SEED,
    deterministic: bool = True,
) -> tuple[StandardScaler, ReturnLSTM]:
    dev = torch.device(device)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    previous_rng_state = torch.get_rng_state()
    try:
        if deterministic:
            deterministic_mode(True, seed=seed)
        else:
            torch.manual_seed(seed)
            torch.set_num_threads(CPU_THREADS)

        features, _x, _y = _train_rows(history)
        next_return = np.log(history["close"]).diff().shift(-1)
        valid = features[list(FEATURES)].notna().all(axis=1) & next_return.notna()
        valid &= history.index.to_numpy() < np.datetime64(TRAIN_END)
        full = features[list(FEATURES)].to_numpy(dtype=float)
        targets = next_return.to_numpy(dtype=float)
        scaler = StandardScaler().fit(full[valid.to_numpy()])
        scaled = scaler.transform(full)

        valid_indices = np.flatnonzero(valid.to_numpy())
        windows = []
        labels = []
        for i in valid_indices:
            if i < WINDOW - 1:
                continue
            window = scaled[i - WINDOW + 1 : i + 1]
            if not np.isfinite(window).all() or not np.isfinite(targets[i]):
                continue
            windows.append(window)
            labels.append(targets[i])
        x = torch.tensor(np.asarray(windows), dtype=torch.float32, device=dev)
        y = torch.tensor(np.asarray(labels), dtype=torch.float32, device=dev)

        model = ReturnLSTM(len(FEATURES), 32).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        model.train()
        for _epoch in range(60):
            permutation = torch.randperm(x.shape[0], generator=generator).to(dev)
            for start in range(0, x.shape[0], 64):
                batch = permutation[start : start + 64]
                optimizer.zero_grad()
                loss = loss_fn(model(x[batch]), y[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        return scaler, model
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)
        torch.set_rng_state(previous_rng_state)


def _run_csco_benchmark(
    history: pd.DataFrame | None = None,
    device: str = "cpu",
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Implement the benchmark while the caller constrains native threads."""
    if history is None:
        history = load_csco_golden_history()

    np.random.seed(seed)
    train_history = history.loc[history.index <= TRAIN_END]
    missing_target_days = [day for day in TARGET_DAYS if day not in history.index]
    if missing_target_days:
        raise FixtureIntegrityError(
            f"CSCO benchmark is missing target sessions: {missing_target_days}"
        )
    if train_history.empty or train_history.index[-1] != TRAIN_END:
        raise FixtureIntegrityError(
            f"CSCO benchmark training history must end on {TRAIN_END.date()}"
        )
    actuals = history.loc[list(TARGET_DAYS), "close"].to_numpy(dtype=float)

    last_close = float(train_history["close"].iloc[-1])
    drift = float(np.log(train_history["close"]).diff().dropna().mean())
    last_week = train_history["close"].iloc[-HORIZON:].to_numpy(dtype=float)

    _features, x, y = _train_rows(train_history)
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(x), y)
    boosting = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=3,
        l2_regularization=0.1,
        random_state=seed,
    ).fit(x, y)

    lstm_scaler, lstm_model = train_csco_lstm(train_history, device=device, seed=seed)

    def ridge_step(row: np.ndarray) -> float:
        return float(ridge.predict(scaler.transform(row[np.newaxis, :]))[0])

    def boosting_step(row: np.ndarray) -> float:
        return float(boosting.predict(row[np.newaxis, :])[0])

    dev = torch.device(device)
    closes_lstm = train_history["close"].to_numpy(dtype=float).copy()
    volumes_lstm = train_history["volume"].to_numpy(dtype=float).copy()
    lstm_preds: list[float] = []
    with torch.no_grad():
        for _session in range(HORIZON):
            feat = compute_csco_feature_frame(closes_lstm, volumes_lstm)
            matrix = feat[list(FEATURES)].to_numpy(dtype=float)
            w = torch.tensor(
                lstm_scaler.transform(matrix[-WINDOW:])[np.newaxis, ...],
                dtype=torch.float32,
                device=dev,
            )
            next_lr = float(lstm_model(w).item())
            next_c = float(closes_lstm[-1]) * float(np.exp(next_lr))
            lstm_preds.append(next_c)
            closes_lstm = np.append(closes_lstm, next_c)
            volumes_lstm = np.append(volumes_lstm, volumes_lstm[-1])

    predictions: dict[str, np.ndarray] = {
        "naive_flat": np.full(HORIZON, last_close),
        "drift_random_walk": last_close * np.exp(drift * np.arange(1, HORIZON + 1)),
        "seasonal_naive_5d": last_week,
        "ridge_lagged_returns": _recursive_forecast("ridge", train_history, ridge_step),
        "hgb_boosting": _recursive_forecast("boosting", train_history, boosting_step),
        "lstm_window": np.asarray(lstm_preds, dtype=float),
    }

    rows = []
    strategies = {
        "naive_flat": "persistence (last close)",
        "drift_random_walk": "random walk with drift",
        "seasonal_naive_5d": "last-week close repeat",
        "ridge_lagged_returns": "ridge on causal lag features",
        "hgb_boosting": "gradient boosting on causal lag features",
        "lstm_window": f"LSTM on {WINDOW}-session windows ({device})",
    }

    for name, pred in predictions.items():
        error = pred - actuals
        rows.append(
            {
                "model": name,
                "strategy": strategies[name],
                "mape_pct": float(np.mean(np.abs(error) / actuals) * 100.0),
                "rmse": float(np.sqrt(mean_squared_error(actuals, pred))),
                "mae": float(mean_absolute_error(actuals, pred)),
                "r2": float(r2_score(actuals, pred)),
            }
        )

    table = pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)
    return table, predictions


def run_csco_benchmark(
    history: pd.DataFrame | None = None,
    device: str = "cpu",
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run the deterministic benchmark with bounded native thread pools.

    Scikit-learn and BLAS backends may otherwise create a platform-dependent
    number of workers. The golden case intentionally uses one native thread so
    it remains viable in constrained CI runners and comparable across runs.
    """
    with threadpool_limits(limits=CPU_THREADS):
        return _run_csco_benchmark(history=history, device=device, seed=seed)


def generate_synthetic_csco_fixture(
    output_path: Path | None = None,
    generator_seed: int = 20260828,
) -> pd.DataFrame:
    """Generate the deterministic synthetic CSCO fixture from mathematical simulation.

    Produces exactly 2,906 NYSE trading sessions from 2015-01-02 to 2026-07-24
    with cyclical macro trends, time-varying volatility regimes, and volume dynamics.
    Unencumbered by external data licenses.
    """
    import exchange_calendars as ecals

    cal = ecals.get_calendar("XNYS")
    sessions = [
        pd.Timestamp(s).normalize() for s in cal.sessions_in_range("2015-01-02", "2026-07-24")
    ]
    n = len(sessions)

    rng = np.random.default_rng(generator_seed)
    s0 = 27.50
    t = np.linspace(0, 11.5, n)
    macro_trend = 0.08 * t + 0.15 * np.sin(2 * np.pi * t / 4.0) + 0.05 * np.cos(2 * np.pi * t / 1.5)
    vol_regime = (
        0.010
        + 0.008 * np.sin(2 * np.pi * t / 3.0) ** 2
        + 0.012 * (np.exp(-((t - 5.2) ** 2) / 0.15) + np.exp(-((t - 9.1) ** 2) / 0.2))
    )

    innovations = rng.standard_normal(n)
    log_returns = np.zeros(n)
    log_returns[0] = 0.0005
    for i in range(1, n):
        drift_step = macro_trend[i] - macro_trend[i - 1]
        ar_term = -0.05 * log_returns[i - 1]
        log_returns[i] = drift_step + ar_term + vol_regime[i] * innovations[i]

    prices = s0 * np.exp(np.cumsum(log_returns))
    prices = np.round(prices, 2)
    prices = np.maximum(prices, 1.0)

    base_vol = 25_000_000
    vol_multiplier = np.exp(rng.normal(0, 0.3, n) + 15.0 * np.abs(log_returns))
    volume = (base_vol * vol_multiplier).astype(int)
    volume = np.maximum(volume, 100_000)

    df = pd.DataFrame(
        {"close": prices, "volume": volume},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target)
    return df
