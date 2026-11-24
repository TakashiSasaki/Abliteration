#!/usr/bin/env python3
"""10-trial Optuna local search for single-layer (layer 25) ARA abliteration.

Fixes start_layer_index=25, end_layer_index=26, caches ARA module I/O to disk,
enqueues the coder3101 baseline point and a 10x steer_bad probe, and runs
Optuna multi-objective optimization (minimize Refusals, minimize KL divergence).
"""
import argparse
import json
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import torch

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model
from heretic.modifiers.ara import ARA, Parameters as ARAParameters, Settings as ARASettings
from heretic.plugin import Context
from heretic.scorer import load_and_init_scorers


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
    print(f"Saved ARA cache to {cache_path} ({size_mb:.1f} MB)")


def load_ara_cache(modifier: ARA, cache_path: Path) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    modifier.good_module_io = payload["good_module_io"]
    modifier.bad_module_io = payload["bad_module_io"]
    print(f"Loaded ARA cache from {cache_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-path", default="/content/cache/gemma4_e4b_bnb4bit_seed42_ara.pt")
    parser.add_argument("--work-dir", default="/content/layer25_optuna")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = work_dir / "layer25_optuna_journal.jsonl"
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

    scorers = load_and_init_scorers(settings, model)
    modifier = ARA(ARASettings(), settings)

    cache_path = Path(args.cache_path)
    if cache_path.exists():
        load_ara_cache(modifier, cache_path)
    else:
        t0 = time.perf_counter()
        modifier.init(ctx)
        print(f"ARA init took {time.perf_counter() - t0:.1f}s")
        save_ara_cache(modifier, cache_path)

    model.apply_lora(modifier.settings.lora_rank)

    storage = JournalStorage(JournalFileBackend(str(ckpt_file)))
    study = optuna.create_study(
        study_name="layer25_local_search",
        storage=storage,
        directions=["minimize", "minimize"],
        sampler=TPESampler(seed=42, n_startup_trials=2),
    )

    # Trial 0: coder3101 published weights on layer 25 only
    study.enqueue_trial({
        "preserve_good_behavior_weight": 0.3873,
        "steer_bad_behavior_weight": 0.0003,
        "overcorrect_relative_weight": 0.8555,
        "neighbor_count": 14,
    })
    # Trial 1: 10x stronger steer_bad_behavior_weight probe
    study.enqueue_trial({
        "preserve_good_behavior_weight": 0.3873,
        "steer_bad_behavior_weight": 0.0030,
        "overcorrect_relative_weight": 0.8555,
        "neighbor_count": 14,
    })

    records = []

    def objective(trial: optuna.Trial):
        t0 = time.perf_counter()
        preserve_good = trial.suggest_float("preserve_good_behavior_weight", 0.05, 0.80)
        steer_bad = trial.suggest_float("steer_bad_behavior_weight", 1e-4, 5e-2, log=True)
        overcorrect = trial.suggest_float("overcorrect_relative_weight", 0.30, 1.50)
        neighbor_count = trial.suggest_int("neighbor_count", 5, 25)

        params = ARAParameters(
            start_layer_index=25,
            end_layer_index=26,
            preserve_good_behavior_weight=preserve_good,
            steer_bad_behavior_weight=steer_bad,
            overcorrect_relative_weight=overcorrect,
            neighbor_count=neighbor_count,
        )
        if trial.number > 0:
            modifier.reset_model(ctx)
        modifier.modify_model(ctx, params)

        scores = {}
        for entry in scorers:
            res = entry.scorer.score(ctx)
            scores[entry.name] = float(res.value if hasattr(res, "value") else res)

        refusals = scores.get("Refusals")
        kl_div = scores.get("KL divergence")
        elapsed = round(time.perf_counter() - t0, 2)

        rec = {
            "trial": trial.number,
            "params": params.to_dict(),
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
    print("All trials completed!", flush=True)


if __name__ == "__main__":
    main()
