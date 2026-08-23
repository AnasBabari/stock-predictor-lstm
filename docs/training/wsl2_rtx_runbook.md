# WSL2 & NVIDIA RTX 2060 (6 GB) Global Model Training Runbook

This runbook specifies the hardware, environment setup, and execution workflow for training and certifying the global multi-asset models under WSL2 with GPU acceleration.

---

## 1. Hardware & System Constraints

- **GPU**: NVIDIA GeForce RTX 2060 (6 GB VRAM).
- **Host OS**: Windows 11 / Windows 10 (21H2+).
- **Linux Environment**: Ubuntu 22.04 LTS or 24.04 LTS under **WSL2**.
- **Free Disk Space Requirement**: At least **80–100 GB** recommended under WSL2 native filesystem (`~/stocklstm-runs`, `~/stocklstm-data`).
- **Memory Footprint**:
  - Windows desktop baseline VRAM usage: ~1.0–1.5 GB.
  - Available training VRAM: ~4.5–5.0 GB.
  - Close background GPU-heavy processes (games, Wallpaper Engine, high-res video players) before long training runs.

---

## 2. WSL2 Environment Setup

### 2.1 Update WSL2 and Install Ubuntu
From an elevated Windows PowerShell prompt:
```powershell
wsl --update
wsl --install -d Ubuntu
wsl --shutdown
```

> [!IMPORTANT]
> **Do not install a Linux NVIDIA display driver inside WSL.** The Windows host driver automatically provides CUDA support to WSL2 via the DirectX / CUDA interop driver.

Verify inside Ubuntu terminal:
```bash
nvidia-smi
```
The RTX 2060 should be listed with ~6 GB total VRAM.

---

### 2.2 Clone Repository & Set Up Python 3.11 with `uv`

Inside Ubuntu (using Linux-native filesystem, e.g. `~`):
```bash
cd ~
git clone https://github.com/AnasBabari/stock-predictor-lstm.git
cd stock-predictor-lstm
git checkout main
```

Install `uv` and create Python 3.11 virtual environment:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

uv python install 3.11
uv sync --project backend --frozen --group dev --group training
```

Install TensorFlow with CUDA support:
```bash
uv pip install --python backend/.venv/bin/python "tensorflow[and-cuda]>=2.16,<3"
```

---

### 2.3 Verify TensorFlow GPU Device
Run the verification check inside Ubuntu:
```bash
backend/.venv/bin/python - <<'PY'
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)
if not gpus:
    raise SystemExit("ERROR: No GPU detected by TensorFlow.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    c = tf.matmul(a, b)
    print("GPU MatMul Verification Mean:", float(tf.reduce_mean(c)))
PY
```

---

## 3. RTX 2060 Safe Execution Guidelines

1. **Batch Size**: Default to `batch_size = 32`. Profile memory before attempting `batch_size = 64`.
2. **Memory Growth**: Always enable dynamic GPU memory growth (`set_memory_growth`) before initializing Keras models.
3. **Process Isolation**: Train one neural process at a time.
4. **Session Clearing**: Execute `tf.keras.backend.clear_session()` and dispose tensors after every completed fold and seed.
5. **Precision**: Use standard `float32` as reference. Do not activate `mixed_float16` without prior loss-scaling verification.

---

## 4. Pipeline Execution Workflow

### Stage 1: Market Panel Acquisition & Snapshot
```bash
export PANEL_LICENSE_ACKNOWLEDGED=true
backend/.venv/bin/python scripts/build_panel.py \
  --universe-file configs/us_liquid_500.txt \
  --out-dir ~/stocklstm-data/panel-v1 \
  --years 8
```

### Stage 2: Feature Construction & Folds
```bash
backend/.venv/bin/python scripts/run_global_pipeline.py \
  --config configs/global-v1-development.json \
  --panel-dir ~/stocklstm-data/panel-v1 \
  --run-dir ~/stocklstm-runs/global-v1 \
  --mode development \
  --stage folds
```

### Stage 3: Baseline & Full Candidate Evaluation
```bash
backend/.venv/bin/python scripts/run_global_pipeline.py \
  --config configs/global-v1-development.json \
  --panel-dir ~/stocklstm-data/panel-v1 \
  --run-dir ~/stocklstm-runs/global-v1 \
  --mode development \
  --stage evaluate
```

### Stage 4: Champion Selection
```bash
backend/.venv/bin/python scripts/run_global_pipeline.py \
  --config configs/global-v1-development.json \
  --panel-dir ~/stocklstm-data/panel-v1 \
  --run-dir ~/stocklstm-runs/global-v1 \
  --mode development \
  --stage select
```

### Stage 5: Locked Certification Holdout
```bash
backend/.venv/bin/python scripts/run_global_pipeline.py \
  --config configs/global-v1-frozen.json \
  --panel-dir ~/stocklstm-data/panel-v1 \
  --run-dir ~/stocklstm-runs/global-v1 \
  --mode certification \
  --stage certify \
  --open-locked-certification-holdout
```

### Stage 6: Full Refit, Conversion & Signed Release
```bash
backend/.venv/bin/python scripts/run_global_pipeline.py \
  --config configs/global-v1-release.json \
  --panel-dir ~/stocklstm-data/panel-v1 \
  --run-dir ~/stocklstm-runs/global-v1 \
  --mode release \
  --stage all
```
