#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASETS = {
    "mmmu_pro": {
        "aliases": ["mmmu_pro_vision", "mmmu_pro_vision_cot_reasoning"],
        "path": "MMMU/MMMU_Pro",
        "name": "vision",
        "split": "test",
    },
    "ai2d": {
        "aliases": ["ai2d", "ai2d_reasoning"],
        "path": "lmms-lab/ai2d",
        "name": None,
        "split": "test",
    },
    "mathvision_testmini": {
        "aliases": ["mathvision_reason_testmini", "mathvision_reason_testmini_reasoning"],
        "path": "MathLLMs/MathVision",
        "name": None,
        "split": "testmini",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mmmu-pro-size", type=int, default=250)
    parser.add_argument("--ai2d-size", type=int, default=250)
    parser.add_argument("--mathvision-size", type=int, default=0)
    return parser.parse_args()


def dataset_len(spec):
    kwargs = {"split": spec["split"]}
    if spec["name"]:
        ds = load_dataset(spec["path"], spec["name"], **kwargs)
    else:
        ds = load_dataset(spec["path"], **kwargs)
    return len(ds)


def choose_indices(n, size, seed, key):
    if not size or size >= n:
        return list(range(n))
    rng = random.Random(f"{seed}:{key}")
    return sorted(rng.sample(range(n), size))


def main():
    args = parse_args()
    sizes = {
        "mmmu_pro": args.mmmu_pro_size,
        "ai2d": args.ai2d_size,
        "mathvision_testmini": args.mathvision_size,
    }
    payload = {"seed": args.seed, "datasets": {}, "task_indices": {}}
    for key, spec in DATASETS.items():
        n = dataset_len(spec)
        indices = choose_indices(n, sizes[key], args.seed, key)
        payload["datasets"][key] = {"num_rows": n, "selected_count": len(indices), "indices": indices}
        for alias in spec["aliases"]:
            payload["task_indices"][alias] = indices
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] wrote fixed subsamples: {out}")
    for key, item in payload["datasets"].items():
        print(f"[INFO] {key}: {item['selected_count']}/{item['num_rows']}")


if __name__ == "__main__":
    main()
