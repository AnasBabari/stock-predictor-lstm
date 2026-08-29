# StockLSTM Volatility V10 Preregistration Document

## 1. Scientific Protocol Identity
- **Protocol ID:** `volatility-v10`
- **Protocol Status:** Frozen
- **Target Contract:** `future-rv-total-v2`
- **Feature Schema:** `deployable-schema-v5` (26 features)
- **Split Contract:** `unique-origin-70-15-15-v2`

---

## 2. Executive Principles & Historical Disclosure
1. **Informed by V9 Diagnostics:** V10 design choices (independent per-horizon candidate selection and multi-horizon baseline comparisons) were directly informed by V9 development diagnostics. No V9 diagnostic results are reused as V10 certification evidence.
2. **Fail-Closed Data Eligibility:** Certification strictly requires `universe_certification_eligible=true` and `market_panel_certification_eligible=true`. Unverified `yfinance` data stops at the gate and prevents certification.
3. **Single-Use Sealed Test Partition:** The 15% sealed test partition is opened exactly once per preregistered protocol.

---

## 3. Mathematical Target & Metric Contracts
$$\text{daily\_total\_variance}_t = \log\left(\frac{\text{Open}_t}{\text{Close}_{t-1}}\right)^2 + \max(\text{RogersSatchell}_t, 0)$$
$$V(t, h) = \sum_{k=1}^h \text{daily\_total\_variance}_{t+k}$$
$$z(t, h) = \log(V(t,h) + \epsilon) - \log(B(t,h) + \epsilon)$$
$$\text{QLIKE}(\hat{V}, V) = \frac{V}{\hat{V}} - \log\left(\frac{V}{\hat{V}}\right) - 1$$
$$\text{Display Annualized Volatility} = \sqrt{\frac{252}{h} \hat{V}}$$

---

## 4. Terminal Outcomes
- **Outcome A:** Learned candidate clears per-horizon gates and is exported to signed ONNX release.
- **Outcome B:** Certified baseline is deployed with published negative result.
- **Outcome C:** Production retains explicit `503 Service Unavailable (abstain_no_certified_model)`.
