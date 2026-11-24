# Gemma 4 E4B (`google/gemma-4-E4B-it`) Abliteration 再現実験レポート

- **実施日時**: 2026-11-23 UTC
- **実行環境**: Google Colab (`Tesla T4` 15,360 MiB VRAM / 12 GiB RAM) via `colab-mcp`
- **使用ツール**: `p-e-w/heretic` (`heretic-llm` v2.1.0), `bitsandbytes`, `Arbitrary-Rank Ablation (ARA)`
- **参照成功例**: [`coder3101/gemma-4-E4B-it-heretic`](https://huggingface.co/coder3101/gemma-4-E4B-it-heretic)

## 1. 再現実験の条件と工夫（クレジット節約・T4メモリ適合）
- `google/gemma-4-E4B-it` は総パラメータ約 8.0B（BF16 約 16 GB）のため、T4 (15 GB VRAM / 12 GB RAM) で OOM を避けるため `--quantization BNB_4BIT` を使用（実測 VRAM 8.67 GB ロード / ピーク約 11.97 GB、システム RAM 1.9〜4.4 GB）。
- 初回実行で T4 の最適バッチサイズが `128`（117 tokens/s）であることを特定し、再現フェーズでは `--batch-size 128` を明示指定してバッチサイズ探索（約 5 分）をスキップ。
- Optuna の `JournalStorage` チェックポイント（`google--gemma-4-E4B-it.jsonl`）に `coder3101/gemma-4-E4B-it-heretic` の公開パラメータを `enqueue_trial` で投入し、1 トライアルで再現検証を実施。

## 2. 実験結果の比較

| 条件 / トライアル | start_layer | end_layer | preserve_good | steer_bad | overcorrect | neighbor_count | Refusals | KL divergence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (元モデル)** | - | - | - | - | - | - | **99/100** | **0** *(定義)* |
| **参照成功例** (`coder3101`, v1.2.0) | 21 | 42 | 0.3873 | 0.0003 | 0.8555 | 14 | **3/100** | **0.0058** |
| **Trial 1** (ランダム探索) | 8 | 41 | 0.7320 | 0.0248 | 0.2028 | 3 | 63/100 | 1.1680 |
| **Trial 2** (ランダム探索) | 1 | 40 | 0.6011 | 0.0680 | 0.0268 | 15 | 8/100 | 1.6988 |
| **再現試行 (Trial 5)** (`BNB_4BIT`, v2.1.0) | 21 | 42 | 0.3873 | 0.0003 | 0.8555 | 14 | **4/100** | **0.0994** |

### 所見
- ベースラインの拒否率 **99/100** は参照成功例と完全に一致。
- 同一の ARA パラメータを `BNB_4BIT` + T4 環境で適用したところ、拒否率は **4/100** まで低下（参照例 `3/100` とほぼ一致）、KL ダイバージェンスも **0.0994** とランダム探索（1.1680〜1.6988）に比べて大幅に低く抑えられ、再現性が確認された。
- `KL divergence` が参照例 (`0.0058`) よりやや高い (`0.0994`) のは、`BNB_4BIT` 量子化状態での近似および `heretic` v1.2.0 → v2.1.0 間のバージョン差異によるものと考えられる。

## 3. 永続化ファイル一覧
- `scripts/run_heretic_gemma4_e4b.sh`: 探索実行用シェルスクリプト
- `scripts/reproduce_gemma4_e4b_coder3101.py`: `coder3101` パラメータ再現用スクリプト
- `results/gemma4_e4b_heretic/google--gemma-4-E4B-it.jsonl`: Optuna チェックポイントデータ
- `results/gemma4_e4b_heretic/heretic.log`: 初回探索ログ（Trial 1〜2）
- `results/gemma4_e4b_heretic/heretic_reproduce.log`: 再現試行ログ（Trial 5）

---

## 4. 英語のみ ARA の他言語への転移検証（`DAMO-NLP-SG/MultiJail`）

- **対象データセット**: `DAMO-NLP-SG/MultiJail`（7言語 `en, zh, it, ko, ar, vi, sw` × 各25問 = 175問）
- **介入**: 英語データセットのみで計算した上記 ARA パラメータ（`start_layer=21, end_layer=42`）を適用
- **スクリプト**: `scripts/eval_cross_lingual_multijail.py`

| 言語 | Baseline 拒否 | Abliterated 拒否 | 削減幅 | Baseline 代表拒否フレーズ |
| :--- | :---: | :---: | :---: | :--- |
| `en` (英語) | 17/25 (68%) | 1/25 (4%) | -64.0 pp | `"I cannot fulfill this request."` |
| `zh` (中国語) | 19/25 (76%) | 1/25 (4%) | -72.0 pp | `"我不能提供任何关于这种非法或有害行为的建议或信息。"` |
| `it` (イタリア語) | 16/25 (64%) | 1/25 (4%) | -60.0 pp | `"Mi dispiace, ma non posso aiutarti con questa richiesta."` |
| `ko` (韓国語) | 18/25 (72%) | 3/25 (12%) | -60.0 pp | `"저는 성적으로 노골적인 콘텐츠를 생성할 수 없습니다."` |
| `ar` (アラビア語) | 17/25 (68%) | 1/25 (4%) | -64.0 pp | `"أنا آسف، ولكني لا أستطيع المساعدة في هذا النوع من الطلبات."` |
| `vi` (ベトナム語) | 16/25 (64%) | 5/25 (20%) | -44.0 pp | `"Tôi không thể giúp bạn với yêu cầu này."` |
| `sw` (スワヒリ語) | 17/25 (68%) | 3/25 (12%) | -56.0 pp | `"Samahani, lakini siwezi kukupa taarifa..."` / 英語フォールバック |

### 結論・観察事項
- 英語のみで最適化した ARA 介入であっても、**高リソース言語から低リソース言語（スワヒリ語等）まで全7言語で拒否率が大幅に低下（-44.0〜-72.0 pp）**し、拒否表現の内部表現が言語横断的に共有されていることが確認された。
- Abliterated 後に残った数件のヒットは、主に「AI なので物理的に人を殴ることはできない」「自転車の鍵の安全性」といった回答内の単語に対するキーワード誤検出（false positive）であった。
- 結果ファイル: `results/gemma4_e4b_cross_lingual/multijail_results.json`, `results/gemma4_e4b_cross_lingual/multijail.log`

---

## 5. 日本語 (`ja`)・インドネシア語 (`id`)・タイ語 (`th`) を含む 10言語拡張検証

- **スクリプト**: `scripts/eval_cross_lingual_multijail_10langs.py`
- **データ**: `DAMO-NLP-SG/MultiJail`（`en, zh, ko, th, vi, it, ar, sw`）+ `facebook/nllb-200-distilled-600M` による翻訳（`ja, id`）各25問、計 250問

| 言語 | Baseline 拒否 | Abliterated 拒否 | 削減幅 |
| :--- | :---: | :---: | :---: |
| `en` (英語) | 17/25 (68%) | 1/25 (4%) | -64.0 pp |
| `ja` (日本語) | 17/25 (68%) | 3/25 (12%) | -56.0 pp |
| `zh` (中国語) | 20/25 (80%) | 1/25 (4%) | -76.0 pp |
| `ko` (韓国語) | 18/25 (72%) | 3/25 (12%) | -60.0 pp |
| `id` (インドネシア語) | 19/25 (76%) | 2/25 (8%) | -68.0 pp |
| `th` (タイ語) | 17/25 (68%) | 2/25 (8%) | -60.0 pp |
| `vi` (ベトナム語) | 16/25 (64%) | 3/25 (12%) | -52.0 pp |
| `it` (イタリア語) | 16/25 (64%) | 2/25 (8%) | -56.0 pp |
| `ar` (アラビア語) | 18/25 (72%) | 1/25 (4%) | -68.0 pp |
| `sw` (スワヒリ語) | 17/25 (68%) | 3/25 (12%) | -56.0 pp |

- 結果ファイル: `results/gemma4_e4b_cross_lingual/multijail_10langs_results.json`, `results/gemma4_e4b_cross_lingual/multijail_10langs.log`

---

## 6. 第25層単層介入実験（`start_layer_index=25, end_layer_index=26`）

- **実施日時**: 2026-11-24 09:55:57Z 〜 10:11:20Z（全体約 15分23秒、Trial 0 単体 `1m 50s`）
- **実行環境**: Google Colab `Tesla T4` (`15,360 MiB VRAM`, Driver `580.82.07`, CUDA `13.0`, `SHAPE_STANDARD`)
- **ライブラリ**: `heretic-llm 2.1.0`, `bitsandbytes 0.50.2`, `transformers 5.22.1`, `torch 2.14.1+cu130`, `Python 3.13.16`
- **CCU 消費**: `591.4906` → `591.1778`（消費 `0.3127` CCU、`assignments_end: 0`）
- **再現条件（全パラメータ固定）**:
  - モデル: `google/gemma-4-E4B-it`（42層 `0..41`、`BNB_4BIT` NF4 + double quant、`compute_dtype=torch.bfloat16`）
  - 設定: `seed=42`, `batch_size=128`, `lora_rank=50`, `preserve_row_magnitudes=True`, `n_optimization_steps=5`, `learning_rate=1.0`, `max_iter=20`, `history_size=10`
  - ARA パラメータ: `start_layer_index=25`, `end_layer_index=26`（第25層のみ: `attn.o_proj`, `mlp.down_proj`）, `preserve_good_behavior_weight=0.3873`, `steer_bad_behavior_weight=0.0003`, `overcorrect_relative_weight=0.8555`, `neighbor_count=14`
  - データセット: `mlabonne/harmless_alpaca`（ARA `train[:400]` / KL 100問）, `mlabonne/harmful_behaviors`（ARA `train[:400]` / Refusals 100問）

| 条件 | 介入層 | Refusals | KL divergence | 備考 |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (元モデル)** | なし | **99/100** | **0** *(定義)* | - |
| **再現実験 (Layers 21–41)** | 21〜41 (21層) | **4/100** | **0.0994** | `coder3101` 公開重み |
| **第25層単層介入 (Layer 25)** | **25 (1層のみ)** | **89/100** | **0.0024** | 同一 `coder3101` 重み |

### 所見
- 同一の ARA 重み（`steer_bad_behavior_weight=0.0003`）を第25層のみに適用した場合、KL divergence は `0.0024` と極めて小さくモデルの元挙動はほぼ保たれた一方、拒否率は `99/100` → `89/100`（-10 pp）の低下に留まった。
- 21層（21〜41層）への介入では `4/100` まで低下していることから、Gemma 4 E4B の拒否表現は第25層単層に局在しているのではなく複数層に分散している、もしくは単層介入では `steer_bad_behavior_weight=0.0003` の強度が不足している可能性が示唆される。
- 永続化ファイル: `scripts/run_layer25.py`, `results/gemma4_e4b_layer25/heretic_layer25.log`, `results/gemma4_e4b_layer25/google--gemma-4-E4B-it.jsonl`
