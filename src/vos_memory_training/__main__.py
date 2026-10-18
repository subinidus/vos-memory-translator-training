"""Help and metadata commands do not import Torch; tensor work needs Torch."""
import argparse
import json
from pathlib import Path
from .common import require, read_json, read_bound, save_bound, write_json, writer


def parser():
    p = argparse.ArgumentParser(description="Frozen MOSE 90/5/5 split, MOSE/LVOS cached-state MSE training, and validation R2 selection")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("inventory", help="Compare the official video list with RGB directory metadata; caches are not inputs")
    q.add_argument("--dataset", choices=("MOSEv2", "LVOSv2"), required=True)
    q.add_argument("--official-split", choices=("train", "valid"), required=True)
    q.add_argument("--video-root", required=True, type=Path)
    q.add_argument("--official-list", required=True, type=Path)
    q.add_argument("--groups", type=Path, help="Verified video_id:original_group JSON; unverified lineage remains UNKNOWN")
    q.add_argument("--output", required=True, type=Path)
    q = sub.add_parser("freeze", help="Create a frozen split from the official MOSE train inventory, or reuse identical inputs")
    q.add_argument("--inventory", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q.add_argument("--seed", default=7, type=int)
    q = sub.add_parser("plan", help="Dry-run roles, exclusions, and storage locations without reading tensors")
    q.add_argument("--datasets", required=True, type=Path)
    q.add_argument("--split", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q = sub.add_parser("inspection-plan", help="List prior-inspection reuse candidates and new inspection scope without reading tensors")
    q.add_argument("--datasets", required=True, type=Path)
    q.add_argument("--split", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q = sub.add_parser("index", help="Check tensors, SHA hashes, and frame/object/slot alignment on CPU without modifying originals")
    q.add_argument("--datasets", required=True, type=Path)
    q.add_argument("--split", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q.add_argument("--operator-completed", required=True)
    q.add_argument("--stable-seconds", default=60, type=float)
    q.add_argument("--max-new-inspections", default=0, type=int, help="Approved maximum new tensor inspections; default 0 permits prior-evidence reuse only")
    q = sub.add_parser("statistics", help="Stream on CPU to compute train-only RMS and fixed validation-target statistics once")
    q.add_argument("--index", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q.add_argument("--config", type=Path)
    q = sub.add_parser("train", help="Single-host torchrun with 1/2/4 ranks; defaults to CPU smoke; CUDA execution requires explicit approval")
    for n in ("index", "statistics", "config", "output"):
        q.add_argument("--" + n, required=True, type=Path)
    q.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    q.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    q.add_argument("--resume", action="store_true")
    q.add_argument("--smoke-evidence", type=Path)
    q.add_argument("--max-wall-seconds", type=float, required=True)
    q.add_argument("--stop-after-epoch", type=int)
    q.add_argument("--execute-approved", action="store_true")
    q.add_argument("--gpu-uuid", action="append")
    q.add_argument("--deadline-utc")
    q.add_argument("--approval-start-utc")
    q.add_argument("--pod-hourly-rate", type=float)
    q.add_argument("--budget-usd", type=float)
    q = sub.add_parser("monitor", help="Read LIVE_STATUS.json without controlling the run")
    q.add_argument("--run", required=True, type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        for v in vars(args).values():
            require(not isinstance(v, Path) or ("<" not in str(v) and "PLACEHOLDER" not in str(v)), "UNFILLED_PATH_PLACEHOLDER")
        if args.command == "inventory":
            from .splits import inventory
            require(not args.output.resolve().is_relative_to(args.video_root.resolve()), "INVENTORY_OUTPUT_INSIDE_RAW_RGB")
            value = inventory(args.dataset, args.official_split, args.video_root, args.official_list, args.groups)
            with writer(args.output.parent):
                save_bound(args.output, value)
        elif args.command == "freeze":
            from .splits import freeze
            value = freeze(read_bound(args.inventory), args.output, args.seed)
        elif args.command == "plan":
            from .splits import read_split
            from .index import plan
            items, sources, _ = plan(read_json(args.datasets), read_split(args.split))
            value = dict(status="DRY_RUN_NOT_TENSOR_PASS", cases=items, sources=sources)
            write_json(args.output, value, immutable=True)
        elif args.command == 'inspection-plan':
            from .index import inspection_plan
            value = inspection_plan(args.datasets, args.split)
            write_json(args.output, value, immutable=True)
        elif args.command == "index":
            require(__import__("os").environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_INDEX_MUST_HIDE_CUDA")
            from .index import build_index
            value = build_index(args.datasets, args.split, args.output, operator_completed=args.operator_completed, stable_seconds=args.stable_seconds,
                                max_new_inspections=args.max_new_inspections)
        elif args.command == "statistics":
            require(__import__("os").environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_STATS_MUST_HIDE_CUDA")
            from .statistics import prepare
            from .policy import Config
            value = prepare(args.index, args.output, Config(**read_json(args.config)) if args.config else None)
        elif args.command == "train":
            require(args.mode != "smoke" or args.max_wall_seconds <= 600, "SMOKE_MAX_600_SECONDS")
            from .training import train
            train(args)
            return 0
        else:
            value = read_json(args.run / "LIVE_STATUS.json")
        print(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Stopped: {type(exc).__name__}: {exc}")
        return 3 if type(exc).__name__ == "LimitReached" else 2


if __name__ == "__main__":
    raise SystemExit(main())
