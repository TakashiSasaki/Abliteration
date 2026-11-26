---
license: gemma
base_model: google/gemma-4-E4B-it
pipeline_tag: text-generation
library_name: gguf
language:
  - en
  - ja
  - zh
  - ko
  - id
  - th
  - vi
  - it
  - ar
  - sw
tags:
  - gemma4
  - gguf
  - text-only
  - abliteration
  - heretic
  - speculative-decoding
  - mtp
  - llama.cpp
---

# Gemma 4 E4B-it (Text-Only, 5-Layer Selective ARA Abliterated, GGUF + MTP Draft)

Text-only GGUF builds of **`google/gemma-4-E4B-it`** with localized 5-layer Arbitrary-Rank Ablation (ARA) applied in native `BF16` and verified alongside **`Gemma4AssistantForCausalLM` Multi-Token Prediction (MTP)** speculative decoding for **8 GB VRAM GPUs** (targeted at Windows 11 + RTX 4070 8 GB + `llama.cpp`).

- **Hugging Face Repository**: [`TakashiSasaki/gemma-4-E4B-it-text-abliterated-5L-GGUF`](https://huggingface.co/TakashiSasaki/gemma-4-E4B-it-text-abliterated-5L-GGUF)
- **GitHub Research & Reproduction Repository**: [`TakashiSasaki/Abliteration`](https://github.com/TakashiSasaki/Abliteration) ([GitHub README](https://github.com/TakashiSasaki/Abliteration#readme) | [Full Experiment Report](https://github.com/TakashiSasaki/Abliteration/blob/main/EXPERIMENT_REPORT.md) | [Reproduction Script](https://github.com/TakashiSasaki/Abliteration/blob/main/scripts/reproduce_gemma4_e4b_gguf_8gb.sh) | [Model Card Backup](https://github.com/TakashiSasaki/Abliteration/blob/main/docs/HF_MODEL_CARD.md))

---

## Key Features

- **Text-Only Stripped GGUF (`0` vision/audio tensors)**: Converted via `ggml-org/llama.cpp` (`b12751`, `Gemma4Model`) with multimodal vision/audio encoders removed (`666` tensors).
- **MTP Speculative Decoding Support (`1.87x–1.93x` speedup)**: Includes quantized **`mtp-assistant`** draft GGUFs (`Gemma4AssistantModel`, `49` tensors) for `llama-server --spec-type draft-mtp`.
- **Fits Comfortably in 8 GB VRAM**: Uses `~3.56 GiB` at `ctx=8192` and `~3.66 GiB` at `ctx=16384` (`-ctk q8_0 -ctv q8_0 -fa on`), and stays at **`~4.00 GiB` even at `ctx=32768`** (`mix-Q4KM-Q8`) because `llama.cpp` keeps the large PLE embedding table on host RAM rather than saturating VRAM.
- **Minimal Localized Intervention (5 / 42 layers)**: ARA (`heretic-llm` v2.1.0, `QuantizationMethod.NONE` on `BF16`) applied only to layers `[24, 25, 29, 30, 37]` using Optuna-selected **5L Focused Trial 12** (`KL = 0.024441`, English refusals `2/100`), reducing 10-language MultiJail refusal rate from **`72.0%` (`180/250`) to `7.6%` (`19/250`)** while preserving WikiText-2 PPL and reasoning quality.

---

## Included Files

| File | Size | SHA-256 | Recommended Role |
| :--- | :---: | :--- | :--- |
| `gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf` | `5.30 GB` (`5,302,268,608 B`) | `2d038865e9fe767d72087d4933ee2c243ad3c4c7e52fc9e0161baacf6c9d9e3f` | **Primary Recommended** — lowest PPL (`53.96`) & lowest uniform MultiJail refusals (`19/250`) |
| `gemma-4-E4B-it-abliterated-5L-FT12-mix-Q4KM-Q8.gguf` | `5.35 GB` (`5,349,741,248 B`) | `57b776f1c709041d4c8f643e39a40f8a0b63e61ca11247a097fe923d95705c1b` | **Mixed-Precision Variant** — 10 intervened tensors (`blk.{24,25,29,30,37}.{attn_output,ffn_down}.weight`) kept at `Q8_0`; highest MTP acceptance (`40.27%`) |
| `mtp-assistant-Q8_0.gguf` | `98.65 MB` (`98,653,280 B`) | `80291890b359435865beccdbbf1d793293966f51e4f0fb3abf28f18303eac35` | **Recommended MTP Draft Model** (`--model-draft`) |
| `mtp-assistant-Q4_K_M.gguf` | `77.23 MB` (`77,231,200 B`) | `8e2b986d1f434dde39c5fe04d72f557c5396d691ac9ea5c910a1ceaae6fb0dc0` | Compact MTP Draft Model |
| `SHA256SUMS.txt` | — | — | Checksums for all `.gguf` files |

*(Note: `IQ4_XS` variants were also benchmarked but intentionally excluded from release due to severe WikiText-2 PPL degradation `53.96 -> 58.05` and basic arithmetic reasoning failure on the bat-and-ball test.)*

---

## Quick Start (`llama.cpp` `b12751+` on Windows 11 / 8 GB GPU)

Requires a recent `llama.cpp` build (`b12751` or newer) with Gemma 4 + `--spec-type draft-mtp` support.

### PowerShell (Windows 11)

```powershell
.\llama-server.exe `
  -m .\gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf `
  --model-draft .\mtp-assistant-Q8_0.gguf `
  --spec-type draft-mtp `
  --spec-draft-n-max 4 `
  -ngl 999 `
  -ngld 999 `
  --ctx-size 16384 `
  -ctk q8_0 `
  -ctv q8_0 `
  -fa on `
  --jinja `
  --host 127.0.0.1 `
  --port 8080
```

### Verify Checksums

```powershell
Get-FileHash *.gguf -Algorithm SHA256
```

---

## Intervention Recipe (`BF16` Native ARA)

Applied directly to `google/gemma-4-E4B-it` in `BF16` (`QuantizationMethod.NONE`, `batch_size=16`) before GGUF conversion:

- **Target Layers (0-indexed)**: `[24, 25, 29, 30, 37]` (`blk.(24|25|29|30|37).(attn_output|ffn_down).weight`)
- **Parameters (`5L Focused Trial 12`)**:
  - `preserve_good_behavior_weight`: `0.5387`
  - `steer_bad_behavior_weight`: `0.00819`
  - `overcorrect_relative_weight`: `0.9787`
  - `neighbor_count`: `12`
- **Pre-Quantization BF16 English Gate (`heretic` Evaluator, 100 prompts)**:
  - `Refusals`: `0.02` (`2/100`)
  - `KL divergence`: `0.024441`

---

## Benchmarks & Quantization Comparison

Measured on **NVIDIA L4 (24 GB VRAM, Ada Lovelace SM 8.9)** with `llama.cpp b12751` (CUDA 12.8), `mtp-assistant-Q8_0.gguf`, `-fa on`, `ctx=8192`, `n_slots=4`:

| Variant | Size (`MiB`) | `llama-bench` `pp512` / `tg128` | WikiText-2 PPL (`chunks=32`) | Server MTP OFF (`VRAM`) | Server MTP ON (`VRAM`, Draft Acc.) | MultiJail 10-Lang Refusals | Bat-and-Ball Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Base `Q4_K_M` (unmodified) | `5069.49` | `7023.28` / `68.37` | `52.3827 ± 2.73` | `64.22 t/s` (`3466 MiB`) | `113.29 / 124.02 t/s` (`3582 MiB`) | `180/250 (72.0%)` | — |
| BF16 Abliterated (pre-quant) | `14.9 GiB` | — | `55.3953 ± 2.91` | — | — | `24/250 (9.6%)` (`25.33 t/s`) | — |
| **`5L-FT12-Q4_K_M` (shipped)** | `5041.55` | `7363.58` / `68.19` | **`53.9559 ± 2.80`** | `63.69 t/s` (`3440 MiB`) | `119.17 t/s` (**1.87x**, `3556 MiB`, `38.50%`) | **`19/250 (7.6%)`** (`90.71 t/s`) | **Pass (`$0.10`)** |
| **`5L-FT12-mix-Q4KM-Q8` (shipped)** | `5086.83` | `6901.10` / `66.38` | `54.0490 ± 2.82` | `62.07 t/s` (`3484 MiB`) | **`119.91 t/s`** (**1.93x**, `3600 MiB`, **`40.27%`**) | `21/250 (8.4%)` (`90.80 t/s`) | **Pass (`$0.10`)** |
| `5L-FT12-mix-Q4KM-BF16` *(not shipped)* | `~5232` | `6964.03` / `64.10` | `54.0917 ± 2.82` | `59.42 t/s` (`3630 MiB`) | `113.69 t/s` (`1.91x`, `3746 MiB`, `39.87%`) | `19/250 (7.6%)` | **Pass (`$0.10`)** |
| `5L-FT12-IQ4_XS` *(not shipped)* | `4808.47` | `7680.85` / `72.52` | `58.0482 ± 3.04` | `67.39 t/s` (`3206 MiB`) | `121.26 t/s` (`1.80x`, `3322 MiB`, `39.46%`) | `23/250 (9.2%)` | **Fail (`$0.00`)** |

### Extended Context VRAM (`-ctk q8_0 -ctv q8_0 -fa on`, MTP ON)
- `5L-FT12-Q4_K_M` at `ctx=16384`: **`3,660 MiB`** VRAM (`118.47 t/s`, `40.59%` draft acceptance)
- `5L-FT12-mix-Q4KM-Q8` at `ctx=16384`: **`3,704 MiB`** VRAM (`119.06 t/s`, `42.64%` draft acceptance)
- `5L-FT12-mix-Q4KM-Q8` at `ctx=32768`: **`4,002 MiB`** VRAM (`118.57 t/s`, `42.64%` draft acceptance)

*(MultiJail covers 250 prompts across 10 languages: `en, ja, zh, ko, id, th, vi, it, ar, sw`; refusal detection uses keyword heuristics as documented in `scripts/eval_gguf_multijail_10langs.py` in the GitHub repository.)*

---

## 日本語クイックガイド

- **どれを使うべきか**: 迷ったら **`gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf`** + **`mtp-assistant-Q8_0.gguf`** の組み合わせが最もバランス良好です（最小 PPL・最少拒否率・8GB VRAM に余裕で収容）。MTP 採択率や介入5層の精度維持を優先して比較したい場合は **`mix-Q4KM-Q8`** をお使いください。
- **Windows 11 用起動スクリプト**: GitHub リポジトリの [`scripts/run_gemma4_mtp_rtx4070.ps1`](https://github.com/TakashiSasaki/Abliteration/blob/main/scripts/run_gemma4_mtp_rtx4070.ps1) / [`.bat`](https://github.com/TakashiSasaki/Abliteration/blob/main/scripts/run_gemma4_mtp_rtx4070.bat) をそのまま利用できます。

---

## Ethical Use, License & Disclaimer

- **License**: Derived from [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) and distributed under the **Gemma Terms of Use** (`license: gemma`). Downstream users must comply with Google's Gemma license and prohibited-use policies.
- **Research Purpose & Limitations**: This model is a **mechanistic interpretability / refusal-localization research artifact** studying selective 5-layer ARA weight modification, mixed quantization stability, and MTP speculative decoding compatibility. Safety refusal guardrails from the base instruction-tuned model have been intentionally attenuated for research evaluation; this artifact is **not** intended for harmful, illegal, or unsafe deployment, and outputs must be independently verified and moderated according to your use case.
