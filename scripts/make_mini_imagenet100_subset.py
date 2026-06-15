#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


SPLITS = ("train", "val")


@dataclass
class CopiedFile:
    split: str
    class_name: str
    source: str
    destination: str


def read_class_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def select_classes(class_names: list[str], num_classes: int, strategy: str, seed: int) -> list[str]:
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than 0")
    if num_classes > len(class_names):
        raise ValueError(f"num_classes={num_classes} is larger than the available class list size {len(class_names)}")

    if strategy == "first":
        return class_names[:num_classes]

    if strategy == "random":
        rng = random.Random(seed)
        sampled = class_names[:]
        rng.shuffle(sampled)
        return sampled[:num_classes]

    raise ValueError(f"Unsupported class selection strategy: {strategy}")


def list_files(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.is_file()])


def prepare_destination(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {path}. Use --overwrite to replace it.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        os.symlink(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def build_subset(
    source_root: Path,
    output_root: Path,
    class_list_path: Path,
    num_classes: int,
    train_images_per_class: int,
    val_images_per_class: int,
    seed: int,
    selection_strategy: str,
    copy_mode: str,
    overwrite: bool,
) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    class_names = read_class_list(class_list_path)
    selected_classes = select_classes(class_names, num_classes, selection_strategy, seed)

    rng = random.Random(seed)
    copied_files: list[CopiedFile] = []
    per_split_summary: dict[str, dict[str, int]] = {}

    prepare_destination(output_root, overwrite)

    for split in SPLITS:
        source_split_root = source_root / split
        if not source_split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {source_split_root}")

        destination_split_root = output_root / split
        destination_split_root.mkdir(parents=True, exist_ok=True)

        requested_per_class = train_images_per_class if split == "train" else val_images_per_class
        per_split_summary[split] = {
            "requested_per_class": requested_per_class,
            "selected_classes": len(selected_classes),
            "copied_files": 0,
        }

        for class_name in selected_classes:
            source_class_dir = source_split_root / class_name
            if not source_class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {source_class_dir}")

            available_files = list_files(source_class_dir)
            if not available_files:
                raise FileNotFoundError(f"No image files found in {source_class_dir}")

            shuffled_files = available_files[:]
            rng.shuffle(shuffled_files)
            chosen_files = shuffled_files[: min(requested_per_class, len(shuffled_files))]

            destination_class_dir = destination_split_root / class_name
            destination_class_dir.mkdir(parents=True, exist_ok=True)

            for source_file in chosen_files:
                destination_file = destination_class_dir / source_file.name
                copy_file(source_file, destination_file, copy_mode)
                copied_files.append(
                    CopiedFile(
                        split=split,
                        class_name=class_name,
                        source=str(source_file),
                        destination=str(destination_file),
                    )
                )

            per_split_summary[split]["copied_files"] += len(chosen_files)

    selected_classes_path = output_root / "selected_classes.txt"
    selected_classes_path.write_text("\n".join(selected_classes) + "\n", encoding="utf-8")

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "class_list_path": str(class_list_path),
        "num_classes": num_classes,
        "selection_strategy": selection_strategy,
        "seed": seed,
        "train_images_per_class": train_images_per_class,
        "val_images_per_class": val_images_per_class,
        "copy_mode": copy_mode,
        "selected_classes": selected_classes,
        "per_split_summary": per_split_summary,
        "files": [asdict(item) for item in copied_files],
    }
    (output_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Created mini ImageNet100 subset at: {output_root}")
    print(f"Selected classes: {len(selected_classes)}")
    print(f"Train images copied: {per_split_summary['train']['copied_files']}")
    print(f"Val images copied: {per_split_summary['val']['copied_files']}")
    print(f"Selected class list saved to: {selected_classes_path}")
    print(f"Manifest saved to: {output_root / 'subset_manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small ImageNet100 subset by keeping a class subset and copying only a few images per class."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Path to the full ImageNet-style root that contains train/ and val/")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/acpl-ssd10/Synapse-0320/results/base/imagenet100_mini"),
        help="Where to write the mini dataset",
    )
    parser.add_argument(
        "--class-list",
        type=Path,
        default=Path("/acpl-ssd10/Synapse-0320/dataset/imagenet100.txt"),
        help="Class list file used to pick a subset",
    )
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes to keep from the class list")
    parser.add_argument(
        "--train-images-per-class",
        type=int,
        default=8,
        help="Maximum number of train images to copy per class",
    )
    parser.add_argument(
        "--val-images-per-class",
        type=int,
        default=4,
        help="Maximum number of val images to copy per class",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for class/image sampling")
    parser.add_argument(
        "--class-selection",
        choices=("first", "random"),
        default="random",
        help="How to choose classes from imagenet100.txt",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How to materialize files in the output dataset",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_subset(
        source_root=args.source_root,
        output_root=args.output_root,
        class_list_path=args.class_list,
        num_classes=args.num_classes,
        train_images_per_class=args.train_images_per_class,
        val_images_per_class=args.val_images_per_class,
        seed=args.seed,
        selection_strategy=args.class_selection,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()