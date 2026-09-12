# Equity Volatility Research Harness

This research harness provides offline empirical evaluation and feature engineering for causal stock volatility prediction. It does not run in Render, the browser, or the production request path.

## Structure

1. **`volatility_structure/`**:
   - High/Low/Open/Close range estimators: Parkinson, Garman-Klass, Rogers-Satchell, and Yang-Zhang.
   - Asymmetric volatility dynamics & leverage effect features (downside semi-variance, sign interactions).
   - Heterogeneous Autoregressive (HAR-RV) components (daily, weekly, monthly log realized volatility).
   - Cross-sectional panel feature extractors for GPU-trained XGBoost / LightGBM models.

2. **`volatility_forecasting/`**:
   - Strict leakage-controlled validation pipelines with $H$-session purged blackout embargoes.
   - Scale-invariant Quasi-Likelihood (QLIKE) and variance RMSE loss evaluation.
   - Statistical comparators: Rolling standard deviation (60d), GARCH(1,1), EWMA ($\lambda=0.94$), and HAR-RV.

## Production Integration

Promoted panel models are packaged as lightweight ONNX graphs under `backend/volatility_models/` (`g3_h5.onnx`, `g3_h10.onnx`, `g3_h20.onnx`) and served causally by `backend/services/g3_volatility.py` without request-time training or a production GPU.
