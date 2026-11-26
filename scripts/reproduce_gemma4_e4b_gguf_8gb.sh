#!/usr/bin/env bash
# End-to-end reproduction script for Gemma 4 E4B (google/gemma-4-E4B-it):
#   1. Download prebuilt llama.cpp b12751 (CUDA 12.8) + source converter (tag b12751 / commit 532d67426)
#   2. Download google/gemma-4-E4B-it, google/gemma-4-E4B-it-assistant, and WikiText-2 test split
#   3. Convert MTP assistant (F16 -> Q8_0 / Q4_K_M)
#   4. Apply 5-layer [24,25,29,30,37] Focused T12 ARA in native BF16 and merge safetensors
#   5. Convert merged BF16 -> text-only GGUF (0 vision/audio tensors)
#   6. Quantize uniform (Q4_K_M, IQ4_XS) and mixed tensor-selective variants
#   7. Benchmark llama-bench, WikiText-2 PPL, llama-server MTP speed/VRAM, and 10-lang MultiJail
#   Optional (RUN_OPTUNA=1): run 8-trial BF16 Optuna on [24,25,29,30,37]
#
# Requirements: NVIDIA GPU with >= 24 GB VRAM (e.g., Colab L4 or A100) and > 32 GB CPU RAM
# for converting the 5.64 GB per_layer_token_embd.weight tensor. Ensure HF_TOKEN is set for
# gated Gemma 4 downloads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/gemma4_repro}"
LLAMA_TAG="${LLAMA_TAG:-b12751}"
PORT="${PORT:-8089}"
RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
RUN_OPTUNA="${RUN_OPTUNA:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

LLAMA_BIN="${WORK_DIR}/llama_bin"
LLAMA_SRC="${WORK_DIR}/llama_src"
HF_DIR="${WORK_DIR}/hf_models"
GGUF_DIR="${WORK_DIR}/gguf_models"
EVAL_DIR="${WORK_DIR}/evals"
CACHE_DIR="${WORK_DIR}/cache"
export TMPDIR="${WORK_DIR}/tmp_gguf"

mkdir -p "$LLAMA_BIN" "$HF_DIR" "$GGUF_DIR" "$EVAL_DIR" "$CACHE_DIR" "$TMPDIR"

if [ "$SKIP_INSTALL" != "1" ]; then
  echo "=== [1/7] Installing Python dependencies & llama.cpp ${LLAMA_TAG} ==="
  pip install -q -r "${REPO_ROOT}/requirements.txt"
  # Required for peft when merging unquantized BF16 (QuantizationMethod.NONE) on Colab/older torchao images
  pip uninstall -y torchao || true

  curl -fsSL -o "${TMPDIR}/llama.tar.gz" \
    "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"
  curl -fsSL -o "${TMPDIR}/cudart.tar.gz" \
    "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/cudart-llama-${LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"
  tar -xzf "${TMPDIR}/llama.tar.gz" -C "$LLAMA_BIN" --strip-components=1
  tar -xzf "${TMPDIR}/cudart.tar.gz" -C "$LLAMA_BIN" --strip-components=1 || \
    tar -xzf "${TMPDIR}/cudart.tar.gz" -C "$LLAMA_BIN"
  rm -f "${TMPDIR}/llama.tar.gz" "${TMPDIR}/cudart.tar.gz"

  if [ ! -d "$LLAMA_SRC/.git" ]; then
    git clone --branch "$LLAMA_TAG" --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
  else
    git -C "$LLAMA_SRC" fetch --depth 1 origin tag "$LLAMA_TAG"
    git -C "$LLAMA_SRC" checkout "$LLAMA_TAG"
  fi
  pip install -q -e "${LLAMA_SRC}/gguf-py"
fi

export LD_LIBRARY_PATH="${LLAMA_BIN}:${LD_LIBRARY_PATH:-}"

echo "=== [2/7] Downloading Gemma 4 E4B, MTP assistant, and WikiText-2 ==="
HF_DIR_ENV="$HF_DIR" CACHE_DIR_ENV="$CACHE_DIR" python3 - <<'PYEOF'
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

hf_dir = Path(os.environ["HF_DIR_ENV"])
cache_dir = Path(os.environ["CACHE_DIR_ENV"])
snapshot_download("google/gemma-4-E4B-it", local_dir=str(hf_dir / "gemma-4-E4B-it"), ignore_patterns=["*.bin", "*.pt"])
snapshot_download("google/gemma-4-E4B-it-assistant", local_dir=str(hf_dir / "gemma-4-E4B-it-assistant"), ignore_patterns=["*.bin", "*.pt"])

wt_path = cache_dir / "wikitext2_test.txt"
if not wt_path.exists():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_path.write_text("\n".join(x["text"] for x in ds if x["text"].strip()), encoding="utf-8")
PYEOF

python3 "${SCRIPT_DIR}/eval_gguf_multijail_10langs.py" \
  --cache-json "${CACHE_DIR}/multijail_10langs_prompts.json" \
  --prepare-only \
  --output-json "${TMPDIR}/unused.json"

echo "=== [3/7] Converting MTP assistant -> F16 / Q8_0 / Q4_K_M ==="
python3 "${LLAMA_SRC}/convert_hf_to_gguf.py" \
  "${HF_DIR}/gemma-4-E4B-it-assistant" \
  --outtype f16 \
  --outfile "${GGUF_DIR}/mtp-assistant-F16.gguf"
"${LLAMA_BIN}/llama-quantize" "${GGUF_DIR}/mtp-assistant-F16.gguf" "${GGUF_DIR}/mtp-assistant-Q8_0.gguf" Q8_0
"${LLAMA_BIN}/llama-quantize" "${GGUF_DIR}/mtp-assistant-F16.gguf" "${GGUF_DIR}/mtp-assistant-Q4_K_M.gguf" Q4_K_M

echo "=== [4/7] Applying 5L [24,25,29,30,37] Focused T12 ARA in BF16 ==="
python3 "${SCRIPT_DIR}/apply_ara_and_merge.py" \
  --layers "24,25,29,30,37" \
  --preserve-good 0.5387 \
  --steer-bad 0.00819 \
  --overcorrect 0.9787 \
  --neighbor-count 12 \
  --quantization NONE \
  --batch-size 16 \
  --cache-path "${CACHE_DIR}/gemma4_e4b_bf16_seed42_ara.pt" \
  --save-dir "${HF_DIR}/gemma4_e4b_5l_ft12_bf16" \
  --result-json "${EVAL_DIR}/ara_bf16_merge_result.json"

echo "=== [5/7] Converting merged BF16 -> text-only GGUF ==="
python3 "${LLAMA_SRC}/convert_hf_to_gguf.py" \
  "${HF_DIR}/gemma4_e4b_5l_ft12_bf16" \
  --use-temp-file \
  --outtype bf16 \
  --outfile "${GGUF_DIR}/abl-5l-ft12-BF16.gguf"

echo "=== [6/7] Quantizing uniform & tensor-selective mixed GGUFs ==="
NPROC="$(nproc)"
SRC="${GGUF_DIR}/abl-5l-ft12-BF16.gguf"
QBIN="${LLAMA_BIN}/llama-quantize"
PAT='blk\.(24|25|29|30|37)\.(attn_output|ffn_down)\.weight'

"$QBIN" "$SRC" "${GGUF_DIR}/gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf" Q4_K_M "$NPROC"
"$QBIN" --tensor-type "${PAT}=q8_0" "$SRC" "${GGUF_DIR}/gemma-4-E4B-it-abliterated-5L-FT12-mix-Q4KM-Q8.gguf" Q4_K_M "$NPROC"
"$QBIN" --tensor-type "${PAT}=bf16" "$SRC" "${GGUF_DIR}/abl-5l-ft12-mix-Q4KM-BF16.gguf" Q4_K_M "$NPROC"
"$QBIN" "$SRC" "${GGUF_DIR}/abl-5l-ft12-IQ4_XS.gguf" IQ4_XS "$NPROC"
"$QBIN" --tensor-type "${PAT}=q8_0" "$SRC" "${GGUF_DIR}/abl-5l-ft12-mix-IQ4XS-Q8.gguf" IQ4_XS "$NPROC"

(cd "$GGUF_DIR" && sha256sum \
  gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf \
  gemma-4-E4B-it-abliterated-5L-FT12-mix-Q4KM-Q8.gguf \
  mtp-assistant-Q4_K_M.gguf \
  mtp-assistant-Q8_0.gguf > SHA256SUMS.txt)

if [ "$RUN_OPTUNA" = "1" ]; then
  echo "=== [Optional] Running 8-trial BF16 Optuna on [24,25,29,30,37] ==="
  python3 "${SCRIPT_DIR}/run_layer29_24_optuna.py" \
    --layers "24,25,29,30,37" \
    --quantization NONE \
    --batch-size 16 \
    --n-trials 8 \
    --cache-path "${CACHE_DIR}/gemma4_e4b_bf16_seed42_ara.pt" \
    --work-dir "${WORK_DIR}/phase4_5_bf16_optuna"
fi

if [ "$RUN_BENCHMARKS" != "1" ]; then
  echo "RUN_BENCHMARKS=${RUN_BENCHMARKS} -> finished conversion & quantization in ${GGUF_DIR}"
  exit 0
fi

echo "=== [7/7] Benchmarking llama-bench, WikiText-2 PPL, MTP server, and 10-lang MultiJail ==="
DRAFT="${GGUF_DIR}/mtp-assistant-Q8_0.gguf"

run_server_eval() {
  local name="$1"
  local gguf="$2"
  local mode="$3"
  local extra="$4"
  local ctx="$5"
  local do_mj="$6"

  # shellcheck disable=SC2086
  "${LLAMA_BIN}/llama-server" \
    -m "$gguf" -ngl 999 --ctx-size "$ctx" --jinja --host 127.0.0.1 --port "$PORT" \
    $extra > "${EVAL_DIR}/srv_${name}_${mode}.log" 2>&1 &
  local spid=$!
  for _ in $(seq 1 60); do
    if curl -sSf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "llama-server exited early for ${name} ${mode}" >&2
      tail -n 30 "${EVAL_DIR}/srv_${name}_${mode}.log" >&2
      return 1
    fi
    sleep 1
  done

  local idle_vram
  idle_vram="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"

  python3 "${SCRIPT_DIR}/bench_gguf_server.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --output-json "${EVAL_DIR}/speed_${name}_${mode}.json" \
    --idle-vram-mib "$idle_vram"

  local active_vram
  active_vram="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  python3 -c "import json; p='${EVAL_DIR}/speed_${name}_${mode}.json'; d=json.load(open(p)); d['active_vram_mib']=${active_vram}; json.dump(d, open(p,'w'), ensure_ascii=False, indent=2)"

  if [ "$do_mj" = "1" ]; then
    python3 "${SCRIPT_DIR}/eval_gguf_multijail_10langs.py" \
      --base-url "http://127.0.0.1:${PORT}" \
      --cache-json "${CACHE_DIR}/multijail_10langs_prompts.json" \
      --output-json "${EVAL_DIR}/multijail_${name}_${mode}.json" \
      --label "${name}_${mode}" --max-tokens 100 --disable-thinking
  fi

  kill "$spid" || true
  wait "$spid" 2>/dev/null || true
}

VARIANT_NAMES=(
  "abl-5l-ft12-Q4_K_M"
  "abl-5l-ft12-mix-Q4KM-Q8"
  "abl-5l-ft12-mix-Q4KM-BF16"
  "abl-5l-ft12-IQ4_XS"
  "abl-5l-ft12-mix-IQ4XS-Q8"
)
VARIANT_PATHS=(
  "${GGUF_DIR}/gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf"
  "${GGUF_DIR}/gemma-4-E4B-it-abliterated-5L-FT12-mix-Q4KM-Q8.gguf"
  "${GGUF_DIR}/abl-5l-ft12-mix-Q4KM-BF16.gguf"
  "${GGUF_DIR}/abl-5l-ft12-IQ4_XS.gguf"
  "${GGUF_DIR}/abl-5l-ft12-mix-IQ4XS-Q8.gguf"
)

for idx in "${!VARIANT_NAMES[@]}"; do
  v="${VARIANT_NAMES[$idx]}"
  GGUF="${VARIANT_PATHS[$idx]}"
  "${LLAMA_BIN}/llama-bench" -m "$GGUF" -ngl 999 -p 512 -n 128 -r 3 -fa 1 -o json > "${EVAL_DIR}/bench_${v}.json"
  "${LLAMA_BIN}/llama-perplexity" -m "$GGUF" -f "${CACHE_DIR}/wikitext2_test.txt" -ngl 999 -c 512 --chunks 32 -fa on 2>&1 \
    | tee "${EVAL_DIR}/ppl_${v}.txt" | tail -n 5
  run_server_eval "$v" "$GGUF" "mtp_off" "-fa on" 8192 0
  run_server_eval "$v" "$GGUF" "mtp_on" "--model-draft ${DRAFT} --spec-type draft-mtp --spec-draft-n-max 4 -ngld 999 -fa on" 8192 1
done

run_server_eval "abl-5l-ft12-Q4_K_M" "${VARIANT_PATHS[0]}" "mtp_on_ctx16k_kvq8" \
  "--model-draft ${DRAFT} --spec-type draft-mtp --spec-draft-n-max 4 -ngld 999 -ctk q8_0 -ctv q8_0 -fa on" 16384 0
run_server_eval "abl-5l-ft12-mix-Q4KM-Q8" "${VARIANT_PATHS[1]}" "mtp_on_ctx32k_kvq8" \
  "--model-draft ${DRAFT} --spec-type draft-mtp --spec-draft-n-max 4 -ngld 999 -ctk q8_0 -ctv q8_0 -fa on" 32768 0

echo "ALL_DONE -> GGUFs: ${GGUF_DIR} | Benchmarks: ${EVAL_DIR}"
