"""Prepare the compact human-goal WebShop catalogue and Table-3 splits.

The released WebShop catalogue has about 1.2M products.  Skill0.5 evaluates
the 12,087 human goals against a compact catalogue of approximately 50K
products (as documented in its released environment manager). We construct
that catalogue deterministically without loading the 5.5GB JSON file into
memory, then materialize the exact
released train/ID-validation/OOD-validation goal indices as Slime parquet
files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import ijson
import pandas as pd


def classify_goal(text: str) -> str:
    goal = text.lower()
    groups = (
        ("apparel", ("shirt", "dress", "jacket", "pant", "coat", "sweater",
                     "blouse", "clothing", "clothes", "t-shirt")),
        ("footwear", ("shoe", "boot", "sneaker", "sandal", "heel", "slipper",
                      "footwear")),
        ("electronics", ("laptop", "phone", "computer", "tablet", "charger",
                         "cable", "headphone", "speaker", "camera", "electronic")),
        ("accessories", ("necklace", "ring", "bracelet", "earring", "watch",
                         "jewelry", "bag", "purse", "wallet")),
        ("home_decor", ("furniture", "lamp", "curtain", "pillow", "bedding",
                        "decor", "candle", "vase", "rug")),
        ("beauty_health", ("cream", "lotion", "shampoo", "conditioner",
                           "moisturizer", "serum", "makeup", "beauty", "vitamin",
                           "supplement")),
    )
    for category, keywords in groups:
        if any(keyword in goal for keyword in keywords):
            return category
    return "other"


def stream_products(path: Path):
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def build_human_catalogue(data_dir: Path) -> tuple[Path, Path]:
    full_items = data_dir / "items_shuffle.json"
    full_attrs = data_dir / "items_ins_v2.json"
    human_goals = data_dir / "items_human_ins.json"
    human_items = data_dir / "items_shuffle_human.json"
    human_attrs = data_dir / "items_ins_v2_human.json"

    if human_items.exists() and human_attrs.exists():
        return human_items, human_attrs

    goal_map = json.loads(human_goals.read_text())
    goal_asins = set(goal_map)

    # Keep all 10,136 products associated with human goals and fill the
    # remainder from the already-shuffled original catalogue. Original order
    # is preserved, so goal generation has exactly the same indices as the
    # full catalogue while each worker stays within practical memory limits.
    target_count = 50_000
    decoy_budget = target_count - len(goal_asins)
    decoy_asins: set[str] = set()
    for product in stream_products(full_items):
        asin = str(product.get("asin"))
        if asin not in goal_asins and len(decoy_asins) < decoy_budget:
            decoy_asins.add(asin)
        if len(decoy_asins) >= decoy_budget:
            break

    selected_asins: set[str] = set()
    tmp_items = human_items.with_suffix(".json.tmp")
    with tmp_items.open("w") as handle:
        handle.write("[")
        first = True
        for product in stream_products(full_items):
            asin = str(product.get("asin"))
            if asin not in goal_asins and asin not in decoy_asins:
                continue
            if not first:
                handle.write(",")
            json.dump(
                product,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
                default=float,
            )
            first = False
            selected_asins.add(asin)
        handle.write("]")
    os.replace(tmp_items, human_items)

    attrs = json.loads(full_attrs.read_text())
    selected_attrs = {asin: value for asin, value in attrs.items()
                      if asin in selected_asins}
    tmp_attrs = human_attrs.with_suffix(".json.tmp")
    tmp_attrs.write_text(json.dumps(selected_attrs, ensure_ascii=False,
                                    separators=(",", ":")))
    os.replace(tmp_attrs, human_attrs)

    missing = goal_asins - selected_asins
    if missing:
        raise RuntimeError(f"{len(missing)} human-goal products are absent")
    print(
        f"[prepare] human catalogue: {len(selected_asins)} products, "
        f"{len(goal_asins)} human-goal products"
    )
    return human_items, human_attrs


def load_goals(vendor_root: Path, items: Path, attrs: Path) -> list[dict]:
    sys.path.insert(0, str(vendor_root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals

    # WebAgentTextEnv seeds before constructing SimServer/load_products.
    random.seed(0)
    products, _, prices, _ = load_products(
        filepath=str(items),
        attrpath=str(attrs),
        num_products=None,
        human_goals=True,
    )
    random.seed(0)
    goals = get_goals(products, prices, human_goals=True)
    random.seed(0)
    random.shuffle(goals)
    return goals


def frame_for_split(
    goals: list[dict], split_name: str, indices: list[int]
) -> pd.DataFrame:
    rows = []
    for goal_idx in indices:
        goal = goals[goal_idx]
        instruction = str(goal["instruction_text"])
        category = classify_goal(instruction)
        rows.append(
            {
                "id": f"{split_name}_{goal_idx}",
                "question": instruction,
                "prompt": [{"role": "user", "content": instruction}],
                "metadata": {
                    "goal_idx": int(goal_idx),
                    "task_description": instruction,
                    "category": category,
                    "task_type": category,
                    "split": split_name,
                    "asin": str(goal.get("asin", "")),
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    items, attrs = build_human_catalogue(args.data_dir)
    goals = load_goals(args.vendor_root, items, attrs)
    splits = json.loads(args.splits.read_text())
    if len(goals) != int(splits["meta"]["total_goals"]):
        raise RuntimeError(
            f"goal count mismatch: runtime={len(goals)}, "
            f"protocol={splits['meta']['total_goals']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split_name in ("train", "val_id", "val_ood"):
        frame = frame_for_split(
            goals, split_name, splits[split_name]["goal_idxs"]
        )
        actual = frame["metadata"].map(lambda value: value["category"]).value_counts()
        expected = splits[split_name]["by_category"]
        actual_dict = {key: int(value) for key, value in actual.items()}
        if actual_dict != expected:
            raise RuntimeError(
                f"{split_name} category mismatch: {actual_dict} != {expected}"
            )
        frame.to_parquet(args.output_dir / f"{split_name}.parquet", index=False)
        frames[split_name] = frame
        print(f"[prepare] {split_name}: {len(frame)} {actual_dict}")

    pd.concat(frames.values(), ignore_index=True).to_parquet(
        args.output_dir / "all.parquet", index=False
    )
    manifest = {
        "protocol": "Skill0.5 Table 3",
        "total_goals": len(goals),
        "splits": {
            name: {
                "count": len(frame),
                "categories": frame["metadata"]
                .map(lambda value: value["category"])
                .value_counts()
                .to_dict(),
            }
            for name, frame in frames.items()
        },
        "catalogue": {
            "items": str(items),
            "attributes": str(attrs),
            "items_bytes": items.stat().st_size,
            "attributes_bytes": attrs.stat().st_size,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


if __name__ == "__main__":
    main()
