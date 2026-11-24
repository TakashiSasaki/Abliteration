#!/usr/bin/env python3
"""10-trial Optuna local search for the best 2-layer pair: (29, 24).

Enqueues coder3101 weights as Trial 0 and a moderate steering probe as Trial 1,
then explores ARA weights around (29, 24) to minimize Refusals and KL divergence.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from peft import PeftModel
import torch

from heretic.config import QuantizationMethod, Settings
from heretic.evaluator import Evaluator
from heretic.model import Model
from heretic.modifiers.ara import ARA, Parameters as ARAParameters, Settings as ARASettings
from heretic.plugin import Context


def save_ara_cache(modifier: ARA, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "good_module_io": modifier.good_module_io,
            "bad_module_io": modifier.bad_module_io,
        },
        cache_path,
    )
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Saved ARA cache to {cache_path} ({size_mb:.1f} MB)", flush=True)


def load_ara_cache(modifier: ARA, cache_path: Path) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    modifier.good_module_io = payload["good_module_io"]
    modifier.bad_module_io = payload["bad_module_io"]
    print(f"Loaded ARA cache from {cache_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="29,24")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-path", default="/content/cache/gemma4_e4b_bnb4bit_seed42_ara.pt")
    parser.add_argument("--work-dir", default="/content/layer29_24_optuna")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = work_dir / "optuna_journal.jsonl"
    if ckpt_file.exists():
        ckpt_file.unlink()

    settings = Settings(
        model="google/gemma-4-E4B-it",
        quantization=QuantizationMethod.BNB_4BIT,
        batch_size=args.batch_size,
        seed=42,
    )
    model = Model(settings)
    ctx = Context(settings=settings, model=model)

    evaluator = Evaluator(settings, model)
    modifier = ARA(heretic_settings=settings, settings=ARASettings())

    cache_path = Path(args.cache_path)
    if cache_path.exists():
        load_ara_cache(modifier, cache_path)
    else:
        t0 = time.perf_counter()
        modifier.init(ctx)
        print(f"ARA init took {time.perf_counter() - t0:.1f}s", flush=True)
        save_ara_cache(modifier, cache_path)

    if not isinstance(model.model, PeftModel):
        model.apply_lora(modifier.settings.lora_rank)

    storage = JournalStorage(JournalFileBackend(str(ckpt_file)))
    study = optuna.create_study(
        study_name="layer29_24_optuna",
        storage=storage,
        directions=["minimize", "minimize"],
        sampler=TPESampler(seed=42, n_startup_trials=2),
    )

    # Trial 0: coder3101 published weights
    study.enqueue_trial({
        "preserve_good_behavior_weight": 0.3873,
        "steer_bad_behavior_weight": 0.0003,
        "overcorrect_relative_weight": 0.8555,
        "neighbor_count": 14,
    })
    # Trial 1: stronger steering + slightly lower preserve probe
    study.enqueue_trial({
        "preserve_good_behavior_weight": 0.2000,
        "steer_bad_behavior_weight": 0.0010,
        "overcorrect_relative_weight": 0.9500,
        "neighbor_count": 14,
    })

    records = []

    def objective(trial: optuna.Trial):
        t0 = time.perf_counter()
        preserve_good = trial.suggest_float("preserve_good_behavior_weight", 0.05, 0.80)
        steer_bad = trial.suggest_float("steer_bad_behavior_weight", 1e-4, 2e-2, log=True)
        overcorrect = trial.suggest_float("overcorrect_relative_weight", 0.40, 1.30)
        neighbor_count = trial.suggest_int("neighbor_count", 8, 24)

        if trial.number > 0:
            modifier.reset_model(ctx)

        for layer_idx in target_layers:
            params = ARAParameters(
                start_layer_index=layer_idx,
                end_layer_index=layer_idx + 1,
                preserve_good_behavior_weight=preserve_good,
                steer_bad_behavior_weight=steer_bad,
                overcorrect_relative_weight=overcorrect,
                neighbor_count=neighbor_count,
            )
            modifier.modify_model(ctx, params)

        score_pairs = evaluator.get_scores()
        scores = {name: float(score.value) for name, score in score_pairs}
        refusals = scores.get("Refusals")
        kl_div = scores.get("KL divergence")
        elapsed = round(time.perf_counter() - t0, 2)

        rec = {
            "trial": trial.number,
            "layers": target_layers,
            "params": {
                "preserve_good_behavior_weight": preserve_good,
                "steer_bad_behavior_weight": steer_bad,
                "overcorrect_relative_weight": overcorrect,
                "neighbor_count": neighbor_count,
            },
            "scores": scores,
            "elapsed_sec": elapsed,
        }
        records.append(rec)
        (work_dir / "optuna_results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[Trial {trial.number}] ({elapsed}s) Refusals={refusals} KL={kl_div:.4f} | "
            f"preserve={preserve_good:.4f}, steer={steer_bad:.5f}, over={overcorrect:.4f}, k={neighbor_count}",
            flush=True,
        )
        return refusals, kl_div

    study.optimize(objective, n_trials=args.n_trials)
    print("Optuna 2-layer search completed!", flush=True)


if __name__ == "__main__":
    main()
