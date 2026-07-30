"""Candidate adapters that give Keras models the baseline runner contract.

TensorFlow is imported lazily: CPU-only CI can still exercise baseline research
without allocating a GPU or importing the serving model module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeuralCandidate:
    """Direct-horizon Keras regression adapter used by the shared runner."""

    architecture: str = "lstm"
    epochs: int = 30
    batch_size: int = 32
    patience: int = 5
    seed: int = 42
    name: str | None = None

    def __post_init__(self) -> None:
        if self.architecture not in {"lstm", "gru", "bilstm_attention_regression"}:
            raise ValueError("Unsupported neural regression architecture.")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 0:
            raise ValueError("Neural candidate training settings are invalid.")
        self.name = self.name or self.architecture
        self.model = None
        self.selected_epoch: int | None = None

    def fit(self, features, targets, *, validation_data=None):
        from model import _build_model_for_type, set_reproducibility

        feature_array = np.asarray(features, dtype=float)
        target_array = np.asarray(targets, dtype=float)
        if (
            feature_array.ndim != 3
            or target_array.ndim != 2
            or len(feature_array) != len(target_array)
        ):
            raise ValueError("Neural features and targets must be aligned 3D/2D arrays.")
        set_reproducibility(self.seed)
        self.model = _build_model_for_type(
            self.architecture, target_array.shape[1], feature_array.shape[2]
        )
        callbacks = []
        if validation_data is not None and self.patience:
            import tensorflow as tf

            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience, restore_best_weights=True
                )
            )
        history = self.model.fit(
            feature_array,
            target_array,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False,
            callbacks=callbacks,
        )
        validation_loss = history.history.get("val_loss", [])
        self.selected_epoch = (
            int(np.argmin(validation_loss) + 1)
            if validation_loss
            else len(history.history.get("loss", []))
        )
        return self

    def refit(self, features, targets):
        """Build a fresh model and refit all outer-fold rows for the selected epoch."""

        if self.selected_epoch is None:
            raise ValueError("Candidate must select an epoch before final refitting.")
        from model import _build_model_for_type, set_reproducibility

        feature_array = np.asarray(features, dtype=float)
        target_array = np.asarray(targets, dtype=float)
        set_reproducibility(self.seed)
        self.model = _build_model_for_type(
            self.architecture, target_array.shape[1], feature_array.shape[2]
        )
        self.model.fit(
            feature_array,
            target_array,
            epochs=self.selected_epoch,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False,
        )
        return self

    def predict(self, features) -> np.ndarray:
        if self.model is None:
            raise ValueError("Forecaster must be fitted before prediction.")
        output = self.model.predict(np.asarray(features, dtype=float), verbose=0)
        # Attention architectures expose auxiliary weights as a second output.
        if isinstance(output, (list, tuple)):
            output = output[0]
        return np.asarray(output, dtype=float).reshape(len(features), -1)

    def metadata(self) -> dict:
        return {
            "architecture": self.architecture,
            "target_type": "regression",
            "seed": self.seed,
            "maximum_epochs": self.epochs,
            "selected_epoch": self.selected_epoch,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
