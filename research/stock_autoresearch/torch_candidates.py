"""Optional PyTorch candidates for offline research only."""

from __future__ import annotations

import numpy as np

from .candidates import Candidate


class TorchLSTMCandidate(Candidate):
    name = "torch_lstm"

    def __init__(self, *, hidden_size=32, dropout=0.2, epochs=8, learning_rate=1e-3, seed=0, device=None):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.device_name = device
        self._model = None
        self._torch = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchLSTMCandidate":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

        class Model(nn.Module):
            def __init__(self, features, hidden, drop):
                super().__init__()
                self.lstm = nn.LSTM(features, hidden, batch_first=True)
                self.dropout = nn.Dropout(drop)
                self.head = nn.Linear(hidden, 1)

            def forward(self, values):
                sequence, _ = self.lstm(values)
                return self.head(self.dropout(sequence[:, -1, :])).squeeze(-1)

        model = Model(int(x.shape[-1]), self.hidden_size, self.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.HuberLoss()
        features = torch.as_tensor(x, dtype=torch.float32, device=device)
        targets = torch.as_tensor(y, dtype=torch.float32, device=device)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        self._model, self._torch = model, torch
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None or self._torch is None:
            raise RuntimeError("TorchLSTMCandidate must be fitted before prediction")
        device = next(self._model.parameters()).device
        values = self._torch.as_tensor(x, dtype=self._torch.float32, device=device)
        self._model.eval()
        with self._torch.no_grad():
            output = self._model(values).detach().cpu().numpy()
        return np.asarray(output, dtype=np.float64)

    def describe(self):
        return {"family": self.name, "hidden_size": self.hidden_size, "dropout": self.dropout,
                "epochs": self.epochs, "learning_rate": self.learning_rate, "seed": self.seed,
                "device": self.device_name or "auto"}
