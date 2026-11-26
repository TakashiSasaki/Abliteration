# Abliteration — Gemma 4 E4B Refusal-Circuit Localization & 8 GB GGUF + MTP Release

Reproducible experiments on mechanistic refusal-circuit localization, Bayesian optimization (`p-e-w/heretic` + Optuna ARA), 10-language cross-lingual transfer (`DAMO-NLP-SG/MultiJail`), native `BF16` merging, tensor-selective mixed quantization, and **8 GB VRAM text-only GGUF + MTP (`--spec-type draft-mtp`)** deployment for **`google/gemma-4-E4B-it`**.

- **Prebuilt GGUF Models (Hugging Face Hub)**: [`TakashiSasaki/gemma-4-E4B-it-text-abliterated-5L-GGUF`](https://huggingface.co/TakashiSasaki/gemma-4-E4B-it-text-abliterated-5L-GGUF) *(Note: large `.gguf` binaries and `models/` are excluded via [`.gitignore`](.gitignore) and hosted on Hugging Face.)*
- **Hugging Face Model Card & Checksum Backup (Disaster Recovery)**: [`docs/HF_MODEL_CARD.md`](docs/HF_MODEL_CARD.md) | [`docs/SHA256SUMS.txt`](docs/SHA256SUMS.txt)
- **Full Experiment Log (Sections 1–29)**: [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)

---

## Key Results Summary

### 1. Minimal 5-Layer Localization (`[24, 25, 29, 30, 37]`) & BF16 Merge
Out of all 42 transformer layers in `google/gemma-4-E4B-it`, applying Arbitrary-Rank Ablation (ARA, `heretic-llm==2.1.0`) to just 5 layers (`[24, 25, 29, 30, 37]`, Focused Trial 12: `preserve_good=0.5387`, `steer_bad=0.00819`, `overcorrect=0.9787`, `neighbor_count=12`) directly on `BF16` (`QuantizationMethod.NONE`) achieves:
- **English Gate (`heretic` Evaluator, 100 prompts)**: Refusals **`2/100 (0.02)`**, KL divergence **`0.024441`**.
- **10-Language MultiJail (`250` prompts across `en, ja, zh, ko, id, th, vi, it, ar, sw`)**:
  - Unmodified Base `Q4_K_M`: `180/250 (72.0%)` refusals
  - Abliterated `BF16` pre-quant: `24/250 (9.6%)` refusals
  - Abliterated `Q4_K_M` (`5L-FT12`): **`19/250 (7.6%)`** refusals

### 2. Text-Only GGUF + MTP Speculative Decoding on 8 GB VRAM (`llama.cpp b12751`)
Evaluated on **NVIDIA L4 (24 GB, Ada Lovelace SM 8.9)** with `mtp-assistant-Q8_0.gguf`, `-fa on`, `ctx=8192`, `n_slots=4`:

| Variant | Size (`MiB`) | `llama-bench` `pp512` / `tg128` | WikiText-2 PPL (`chunks=32`) | Server MTP OFF -> ON (`VRAM`, Draft Acc.) | MultiJail 10-Lang | Bat-and-Ball Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base `Q4_K_M` | `5069.49` | `7023.28` / `68.37` | `52.3827 ± 2.73` | `64.22 -> 113.29 / 124.02 t/s` (`3582 MiB`) | `180/250 (72.0%)` | — |
| **`5L-FT12-Q4_K_M`** *(HF shipped)* | `5041.55` | `7363.58` / `68.19` | **`53.9559 ± 2.80`** | `63.69 -> 119.17 t/s` (**1.87x**, `3556 MiB`, `38.50%`) | **`19/250 (7.6%)`** | **Pass (`$0.10`)** |
| **`5L-FT12-mix-Q4KM-Q8`** *(HF shipped)* | `5086.83` | `6901.10` / `66.38` | `54.0490 ± 2.82` | `62.07 -> 119.91 t/s` (**1.93x**, `3600 MiB`, **`40.27%`**) | `21/250 (8.4%)` | **Pass (`$0.10`)** |
| `5L-FT12-mix-Q4KM-BF16` | `~5232` | `6964.03` / `64.10` | `54.0917 ± 2.82` | `59.42 -> 113.69 t/s` (`3746 MiB`, `39.87%`) | `19/250 (7.6%)` | **Pass (`$0.10`)** |
| `5L-FT12-IQ4_XS` | `4808.47` | `7680.85` / `72.52` | `58.0482 ± 3.04` | `67.39 -> 121.26 t/s` (`3322 MiB`, `39.46%`) | `23/250 (9.2%)` | **Fail (`$0.00`)** |
| `5L-FT12-mix-IQ4XS-Q8` | `4890.81` | `7480.15` / `70.91` | `58.8949 ± 3.10` | `65.96 -> 121.01 t/s` (`3404 MiB`, `40.74%`) | **`18/250 (7.2%)`** | **Fail (`$0.00`)** |

- **Long Context on 8 GB VRAM (`-ctk q8_0 -ctv q8_0 -fa on`, MTP ON)**: `5L-FT12-Q4_K_M` (`ctx=16384`) uses **`3,660 MiB`**; `5L-FT12-mix-Q4KM-Q8` uses **`3,704 MiB`** (`ctx=16384`) and **`4,002 MiB`** (`ctx=32768`).
- **Mixed Quantization & Optuna Re-Optimization Findings**: Raising the 10 intervened tensors (`blk.(24|25|29|30|37).(attn_output|ffn_down).weight`) to `Q8_0` recovers `IQ4_XS` MultiJail refusals (`23/250 -> 18/250`) better than re-optimizing ARA toward stronger steering in Optuna (`21/250`, PPL `61.08`), though base `Q4_K_M` already preserves both reasoning (`$0.10`) and low refusals (`19/250`).

---

## Repository Structure

```text
.
├── EXPERIMENT_REPORT.md      # Detailed record of all 29 experiment sections
├── README.md                 # This file (GitHub research & reproduction guide)
├── ccu_history.jsonl         # Colab GPU compute consumption log
├── requirements.txt          # Pinned Python dependencies & torchao workaround notes
├── docs/
│   ├── HF_MODEL_CARD.md      # Synchronized backup of the Hugging Face Model Card (README.md)
│   └── SHA256SUMS.txt        # Checksums for all released GGUF artifacts
├── scripts/
│   ├── reproduce_gemma4_e4b_gguf_8gb.sh  # End-to-end BF16 ARA merge, GGUF conversion, quant & eval
│   ├── apply_ara_and_merge.py            # Apply ARA on specified layers & merge BF16 safetensors
│   ├── bench_gguf_server.py              # Speed, MTP acceptance & sanity-check client
│   ├── eval_gguf_multijail_10langs.py    # 10-language MultiJail benchmark via llama-server
│   ├── run_layer29_24_optuna.py          # Multi-objective Optuna search (BNB_4BIT or NONE/BF16)
│   ├── run_heretic_gemma4_e4b.sh         # Initial Heretic BNB_4BIT baseline search
│   ├── reproduce_gemma4_e4b_coder3101.py # Reference parameter reproduction
│   ├── run_single_layer_sweep.py         # Single-layer localization sweep
│   ├── run_layer25.py                    # Layer 25 evaluation & probes
│   ├── run_layer25_optuna.py             # Layer 25 Optuna search
│   ├── run_layer29_plus1_sweep.py        # 2-layer expansion sweep from layer 29
│   ├── run_layer24_29_plus1_sweep.py     # 3-layer expansion sweep from [24, 29]
│   ├── run_cached_ara_trials.py          # Fast cached ARA evaluation across layer subsets
│   ├── eval_cross_lingual_*.py           # Cross-lingual HF-model MultiJail evaluators
│   ├── run_gemma4_mtp_rtx4070.ps1        # Windows 11 PowerShell launcher (MTP enabled, 16k ctx)
│   └── run_gemma4_mtp_rtx4070.bat        # Windows 11 CMD batch launcher (MTP enabled, 16k ctx)
└── results/                              # Saved Optuna journals, JSON summaries, and logs
```

---

## How to Reproduce the Experiments

### Requirements
- **End-to-End GGUF Reproduction**: Linux NVIDIA GPU with **>= 24 GB VRAM** (e.g., Colab L4 or A100) and **> 32 GB CPU RAM** (`convert_hf_to_gguf.py` requires > 12 GB host RAM due to the `5.64 GB` `per_layer_token_embd.weight` tensor).
- **Hugging Face Access**: Accept the Gemma license on [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) and [`google/gemma-4-E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant), then export `HF_TOKEN`.

### Full Gemma 4 E4B BF16 Merge + GGUF + Mixed Quantization + Benchmark Pipeline
```bash
git clone https://github.com/TakashiSasaki/Abliteration.git
cd Abliteration
export HF_TOKEN="hf_..."
WORK_DIR=/tmp/gemma4_repro bash scripts/reproduce_gemma4_e4b_gguf_8gb.sh
```
Optional flags:
- `RUN_BENCHMARKS=0`: Build and quantize GGUFs only (skip `llama-bench`, PPL, and MultiJail).
- `RUN_OPTUNA=1`: Also run the 8-trial `BF16` Optuna search (`scripts/run_layer29_24_optuna.py --quantization NONE`).

### Layer Localization & Optuna Sweeps
To reproduce the layer discovery phase recorded in [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md):
```bash
pip install -r requirements.txt
bash scripts/run_heretic_gemma4_e4b.sh
python3 scripts/run_single_layer_sweep.py
python3 scripts/run_layer29_24_optuna.py --layers 24,25,29,30,37 --n-trials 15
```

---

## Using Prebuilt GGUFs on Windows 11 (RTX 4070 8 GB)

Instead of rebuilding from scratch, download the prebuilt `.gguf` files from [`TakashiSasaki/gemma-4-E4B-it-text-abliterated-5L-GGUF`](https://huggingface.co/TakashiSasaki/gemma-4-E4B-it-text-abliterated-5L-GGUF) into `./models/`:

```powershell
hf download TakashiSasaki/gemma-4-E4B-it-text-abliterated-5L-GGUF `
  gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf `
  mtp-assistant-Q8_0.gguf `
  SHA256SUMS.txt `
  --local-dir .\models

.\scripts\run_gemma4_mtp_rtx4070.ps1
```

---

## Restoring the Hugging Face Repository (Disaster Recovery)

If the Hugging Face repository is ever deleted or unavailable, all binaries and metadata can be regenerated identically from this GitHub repository:
1. Run `WORK_DIR=./repro RUN_BENCHMARKS=0 bash scripts/reproduce_gemma4_e4b_gguf_8gb.sh`.
2. Verify checksums against [`docs/SHA256SUMS.txt`](docs/SHA256SUMS.txt) (`sha256sum -c`).
3. Upload [`docs/HF_MODEL_CARD.md`](docs/HF_MODEL_CARD.md) as `README.md` together with the regenerated `.gguf` files and `SHA256SUMS.txt` to Hugging Face Hub.
