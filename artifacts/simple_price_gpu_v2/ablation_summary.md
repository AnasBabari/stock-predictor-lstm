# GPU Price Forecasting: News Signal Ablation Report

- **Model Architecture:** `pooled-price-lstm-v2-attention`
- **Device:** NVIDIA GeForce RTX 2060
- **Tickers:** AAPL, GOOGL, MSFT, NVDA, TSLA
- **Test Rows (Untouched 15%):** 2165

## 1. Pooled Test Performance Comparison

| Metric | Baseline (Price-Only) | Challenger (+News) | Absolute Δ | Relative Δ |
| :--- | :---: | :---: | :---: | :---: |
| MAE (%) | 3.5050% | 3.4898% | -0.0152% | -0.43% |
| RMSE (%) | 4.9949% | 4.9892% | -0.0057% | -0.11% |
| Direction Accuracy | 52.30% | 53.93% | +1.63% | — |
| Rel MAE vs Persistence | 1.0015× | 0.9972× | -0.0043× | — |

## 2. Per-Horizon Test Breakdown (Day 1 to 7)

| Horizon | Base MAE | +News MAE | Δ MAE | Base DirAcc | +News DirAcc | Δ DirAcc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Day 1 | 1.7449% | 1.7412% | -0.0037% | 51.5% | 51.3% | -0.2% |
| Day 2 | 2.5349% | 2.5419% | +0.0070% | 51.3% | 52.1% | +0.7% |
| Day 3 | 3.1647% | 3.1686% | +0.0040% | 51.5% | 53.9% | +2.4% |
| Day 4 | 3.7027% | 3.6922% | -0.0105% | 52.9% | 54.2% | +1.3% |
| Day 5 | 4.1324% | 4.1087% | -0.0237% | 53.1% | 54.3% | +1.2% |
| Day 6 | 4.4616% | 4.4376% | -0.0241% | 52.1% | 56.1% | +4.0% |
| Day 7 | 4.7935% | 4.7385% | -0.0551% | 53.7% | 55.7% | +1.9% |

## 3. Per-Ticker Test Summary

| Ticker | Base MAE | +News MAE | Base DirAcc | +News DirAcc |
| :---: | :---: | :---: | :---: | :---: |
| AAPL | 2.8052% | 2.7346% | 48.2% | 55.0% |
| GOOGL | 3.1418% | 3.1423% | 56.0% | 56.9% |
| MSFT | 2.6736% | 2.6219% | 52.7% | 53.9% |
| NVDA | 3.8450% | 3.8540% | 54.3% | 54.8% |
| TSLA | 5.0592% | 5.0963% | 50.3% | 49.0% |

## 4. Empirical Verdict

- **Verdict:** MARGINAL_DIFFERENCE
- **Rationale:** News feature delta: MAE delta -0.0152%, Direction Accuracy delta +1.63%.
