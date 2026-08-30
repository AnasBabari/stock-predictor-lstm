# V11.2 PIT64 diagnostic CUDA run — 2026-08-30

This record describes a **development-only hardware and candidate diagnostic**.
It is not certification evidence and must not be used to assemble, sign, or
deploy a production release.

## Identity and eligibility

- Git SHA: `50f6563a9084bb8550f7cbab986d6b1be919daac`
- Protocol: `stocklstm-volatility-v11.2-numeric-pit64`
- Protocol SHA-256: `3f3b98684264e7c817389117b10252bb33dd321b22a07eeafce1408f398f7b44`
- Universe SHA-256: `c587ae7ddc3d2f20c5d2701b916caec51ce5cccb134d26ba09c2c0295748682f`
- Panel SHA-256: `190ff73b3fa3c8497c18bef8b472079b79bc03eef0831ee469a3308d9b26bb48`
- Split SHA-256: `ec22040c7335ef5442a96301bf81f09990a8cf66d7d822c3283c3afdb45069d2`
- Development comparison SHA-256: `a6c943fa5cbc2160d9d9015bc55c27a5e7d2aae7511839afd3e33e1366047d62`
- Master freeze SHA-256: `263eba5420355eed9e23a4d0af5ef8e60bd863160fc8c035d2c6cea83e54dfff`
- Eligibility: `certification_eligible=false`
- Reason: market history and membership evidence originate from the repository's
  secondary NDX100 development cache rather than a licensed, independently
  attested PIT64 source.

The universe contains 64 securities in eight balanced research strata. Stable
CIKs came from SEC metadata, FIGIs from OpenFIGI, and descriptive industry data
from Nasdaq. Those identity sources do not upgrade the eligibility of the
underlying OHLCV or membership history.

## Dataset and execution

- Market span: 2022-01-03 through 2026-08-27
- Stock-origin observations: 61,210
- Unique origin sessions: 1,022
- Features: 26 ordered `deployable_v5` features over 60 sessions
- Horizons: 1, 3, 5, and 7 sessions
- Split: chronological 70/15/15 with seven-session purge and 30-session embargo
- Sealed diagnostic reserve: 7,439 observations across 124 sessions
- Sealed status after development: `LOCKED_UNOPENED`
- GPU: NVIDIA GeForce RTX 2060, 6 GiB
- PyTorch: 2.11.0+cu128
- Batch size: 256, chronological, no shuffle
- Epoch cap: 15; patience: 4; seeds: 41, 42, 43
- Observed peak CUDA allocation: approximately 2.2 GiB

The first attempt exposed a full-batch activation allocation above the GPU's
capacity. Commit `812734e` replaced the full-panel forward/backward pass with
bounded chronological batches and weighted gradient accumulation. The focused
V11.2 suite passed before the clean-SHA rerun.

## Development validation results

CRPS and QLIKE are lower-is-better. Coverage is the empirical central 80%
Student-t interval coverage.

| Horizon | Selected route | HAR CRPS | M1 CRPS | HAR QLIKE | M1 QLIKE | HAR coverage | M1 coverage | M1 better seeds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | M1 numeric residual | 0.01008351 | 0.01004730 | 0.71742762 | 0.64545822 | 0.8452 | 0.8341 | 3/3 |
| 3 | HAR baseline | 0.01752994 | 0.01746651 | 0.52711812 | 0.52485491 | 0.8452 | 0.8351 | 3/3 |
| 5 | HAR baseline | 0.02303733 | 0.02294751 | 0.45456506 | 0.45464183 | 0.8372 | 0.8265 | 3/3 |
| 7 | HAR baseline | 0.02741789 | 0.02730914 | 0.40139025 | 0.40216460 | 0.8316 | 0.8215 | 3/3 |

At one session, M1's candidate-minus-HAR mean CRPS delta was
`-0.00003629`; its 95% session-block interval was entirely below zero
(`[-0.00006255, -0.00000685]`) and its Holm-adjusted p-value was `0.01040`.
The route therefore passed the frozen family-wise gate.

At 3, 5, and 7 sessions, M1 had a lower point-estimate CRPS in all three seeds,
but the 95% block-bootstrap intervals crossed zero. Their joint Holm-adjusted
p-value was `0.12179`. Five- and seven-session M1 QLIKE was also slightly worse
than HAR. The strict policy therefore retained HAR for those horizons.

HAR itself passed every preregistered adequacy comparison against constant
variance and realized-volatility persistence across all four horizons, with
central 80% coverage inside the required `[0.65, 0.95]` band.

## What this proves—and what it does not

This run proves that the frozen feature, split, candidate, CUDA, evidence, and
routing pipeline executes on the local RTX 2060 without opening the reserve. It
also supplies useful development evidence that the neural residual adds a small,
statistically supported one-day improvement on this secondary panel.

It does **not** prove production generalization. The diagnostic holdout must
remain unopened, because its inputs are not certification-eligible. Production
completion still requires a licensed immutable OHLCV snapshot, independently
attested PIT64 membership/security master, a new dataset encrypted under the
external certification key, a clean development run, pre-unseal audit, one-shot
certification, ONNX parity, signed release assembly, and deployment verification.
