#!/usr/bin/env python3
"""Apply ARA on target layers (default: 5L [24,25,29,30,37] Focused T12), evaluate, merge to BF16, and save HF safetensors."""
import argparse
import json
import sys
import time
from pathlib import Path

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
    parser.add_argument("--layers", default="24,25,29,30,37")
    parser.add_argument("--preserve-good", type=float, default=0.5387)
    parser.add_argument("--steer-bad", type=float, default=0.00819)
    parser.add_argument("--overcorrect", type=float, default=0.9787)
    parser.add_argument("--neighbor-count", type=int, default=12)
    parser.add_argument("--quantization", default="NONE", choices=["NONE", "BNB_4BIT"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-path", default="/content/cache/gemma4_e4b_bf16_seed42_ara.pt")
    parser.add_argument("--save-dir", default="/content/hf_models/gemma4_e4b_5l_ft12_bf16")
    parser.add_argument("--save-adapter-dir", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--result-json", default="/content/ara_bf16_merge_result.json")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    qmethod = QuantizationMethod.NONE if args.quantization == "NONE" else QuantizationMethod.BNB_4BIT

    t0 = time.perf_counter()
    settings = Settings(
        model="google/gemma-4-E4B-it",
        quantization=qmethod,
        batch_size=args.batch_size,
        seed=42,
    )
    model = Model(settings)
    ctx = Context(settings=settings, model=model)
    modifier = ARA(heretic_settings=settings, settings=ARASettings())

    evaluator = None if args.skip_eval else Evaluator(settings, model)

    cache_path = Path(args.cache_path)
    if cache_path.exists():
        load_ara_cache(modifier, cache_path)
    else:
        t_init = time.perf_counter()
        modifier.init(ctx)
        print(f"ARA init took {time.perf_counter() - t_init:.1f}s", flush=True)
        save_ara_cache(modifier, cache_path)

    if not isinstance(model.model, PeftModel):
        model.apply_lora(modifier.settings.lora_rank)

    t_mod = time.perf_counter()
    for layer_idx in target_layers:
        params = ARAParameters(
            start_layer_index=layer_idx,
            end_layer_index=layer_idx + 1,
            preserve_good_behavior_weight=args.preserve_good,
            steer_bad_behavior_weight=args.steer_bad,
            overcorrect_relative_weight=args.overcorrect,
            neighbor_count=args.neighbor_count,
        )
        modifier.modify_model(ctx, params)
    mod_sec = round(time.perf_counter() - t_mod, 2)
    print(f"ARA modify_model took {mod_sec}s across layers {target_layers}", flush=True)

    scores = {}
    if evaluator is not None:
        t_eval = time.perf_counter()
        score_pairs = evaluator.get_scores()
        scores = {name: float(score.value) for name, score in score_pairs}
        print(f"Evaluator ({time.perf_counter()-t_eval:.1f}s): {scores}", flush=True)

    if args.save_adapter_dir:
        ad = Path(args.save_adapter_dir)
        ad.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(ad)
        print(f"Saved PEFT LoRA adapter to {ad}", flush=True)

    # Free evaluator & ARA caches before merging/saving to minimize peak RAM/VRAM
    del evaluator
    modifier.good_module_io.clear()
    modifier.bad_module_io.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    t_merge = time.perf_counter()
    merged = model.get_merged_model()
    merged.save_pretrained(save_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(save_dir)
    if getattr(model, "processor", None) is not None:
        model.processor.save_pretrained(save_dir)
    merge_sec = round(time.perf_counter() - t_merge, 2)

    out = {
        "layers": target_layers,
        "quantization_for_ara": args.quantization,
        "params": {
            "preserve_good_behavior_weight": args.preserve_good,
            "steer_bad_behavior_weight": args.steer_bad,
            "overcorrect_relative_weight": args.overcorrect,
            "neighbor_count": args.neighbor_count,
        },
        "scores": scores,
        "modify_sec": mod_sec,
        "merge_save_sec": merge_sec,
        "total_sec": round(time.perf_counter() - t0, 2),
        "save_dir": str(save_dir),
    }
    Path(args.result_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE:", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
