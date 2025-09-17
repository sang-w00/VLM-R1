#!/usr/bin/env python3
"""Build training JSONL for numbered_square scenes.

Each scene JSON (under --scenes-dir) describes a scene with:
  - objects (one huge gray cube + two small cubes of different colors)
  - views (we only use view_00.png as the input image path)
  - numbers: {"left": int, "right": int}

For every scene we generate TWO QA entries (one asking left number, one asking right number),
using the following prompt template (with randomized color1/color2 order per scene):

Template:
You are given an image showing a specific view of a 3D environment. In this scene, the largest gray cube is placed at the center, surrounded by two smaller cubes. Choose one action from the choices (A or B) below in order to answer a given question about this environment. Actions: A. Look at either side of the gray cube from the {color1} box's perspective B. Look at either side of the gray cube from the {color2} box's perspective The question is as follow: <image> What number is written in the {side} side of the gray cube?

Output JSONL line example:
{"id": 1, "image": "/abs/path/scene_000000/view_00.png", "conversations": [{"from": "human", "value": "<PROMPT_WITH_SIDE_LEFT>"}, {"from": "gpt", "value": "41"}]}

Notes:
  * color1/color2 are sampled as a random permutation of the two small cube colors (per scene, same order used for both left/right questions in that scene).
  * side in {side} is either left or right.
  * answer value is numbers[side].
  * IDs increment sequentially starting from 1 across all produced lines.
  * Deterministic behavior can be achieved with --seed.

Usage:
  python dataset/build_numbered_square_training_jsonl.py \
      --scenes-dir /home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/numbered_square/training/scenes \
      --images-dir /home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/numbered_square/training/images \
      --out-file dataset/numbered_square_training.jsonl

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, Any, List

PROMPT_TEMPLATE = (
    "You are given an image showing a specific view of a 3D environment. In this scene, the largest gray cube is placed at the center, "
    "surrounded by two smaller cubes. Choose one action from the choices (A or B) below in order to answer a given question about this environment. "
    "Actions: A. Look at either side of the gray cube from the {color1} box's perspective B. Look at either side of the gray cube from the {color2} box's perspective "
    "The question is as follow: <image> What number is written in the {side} side of the gray cube?\n"
)


def parse_args():
    ap = argparse.ArgumentParser(description="Generate training JSONL for numbered_square scenes")
    ap.add_argument('--scenes-dir', required=True, help='Directory containing scene_XXXXXX.json files')
    ap.add_argument('--images-dir', required=True, help='Directory containing scene_XXXXXX/view_00.png images')
    ap.add_argument('--out-file', required=True, help='Output JSONL path')
    ap.add_argument('--seed', type=int, default=42, help='Random seed (for color order shuffling)')
    ap.add_argument('--max-scenes', type=int, default=None, help='Optional limit on number of scenes processed')
    ap.add_argument('--verbose', action='store_true', help='Print per-scene debug info')
    ap.add_argument('--glob-pattern', default='*scene_*.json', help='Glob pattern for scene JSON files (default: *scene_*.json)')
    return ap.parse_args()


def load_scene_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def extract_small_cube_colors(scene: Dict[str, Any]) -> List[str]:
    colors = []
    for obj in scene.get('objects', []):
        if obj.get('shape') == 'cube' and obj.get('size') == 'small':
            color = obj.get('color')
            if isinstance(color, str):
                colors.append(color.strip())
    # Deduplicate while preserving order
    dedup = []
    for c in colors:
        if c not in dedup:
            dedup.append(c)
    if len(dedup) != 2:
        raise ValueError(f"Expected exactly 2 small cube colors, got {dedup}")
    return dedup


def build_entries_for_scene(scene_path: Path, images_dir: Path, next_id: int, rng: random.Random, *, verbose: bool = False) -> List[Dict[str, Any]]:
    scene = load_scene_json(scene_path)
    scene_index = scene.get('scene_index')
    # Colors
    small_colors = extract_small_cube_colors(scene)
    rng.shuffle(small_colors)  # in-place shuffle
    color1, color2 = small_colors
    # Numbers
    numbers = scene.get('numbers', {})
    if not all(k in numbers for k in ('left', 'right')):
        raise ValueError(f"Scene {scene_index} missing left/right numbers")
    # Image path (view_00.png)
    scene_dir_name = f"scene_{scene_index:06d}"
    # Auto-detect extension (png or jpg)
    base_dir = images_dir / scene_dir_name
    png_path = base_dir / 'view_00.png'
    jpg_path = base_dir / 'view_00.jpg'
    if png_path.exists():
        image_path = png_path
    elif jpg_path.exists():
        image_path = jpg_path
    else:
        raise FileNotFoundError(f"Image not found (tried): {png_path} and {jpg_path}")

    if verbose:
        print(f"[debug] scene_index={scene_index} colors(shuffled)={small_colors} image={image_path.name}")

    entries = []
    for side in ('left', 'right'):
        prompt = PROMPT_TEMPLATE.format(color1=color1, color2=color2, side=side)
        answer = str(numbers[side])
        entry = {
            'id': next_id,
            'image': str(image_path),
            'conversations': [
                {'from': 'human', 'value': prompt},
                {'from': 'gpt', 'value': answer}
            ]
        }
        entries.append(entry)
        next_id += 1
    return entries


def main():
    args = parse_args()
    scenes_dir = Path(args.scenes_dir)
    images_dir = Path(args.images_dir)
    out_file = Path(args.out_file)

    if not scenes_dir.exists():
        print(f"[error] scenes-dir does not exist: {scenes_dir}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.exists():
        print(f"[error] images-dir does not exist: {images_dir}", file=sys.stderr)
        sys.exit(1)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Support filenames like CLEVR_NumberedSquare_numbered_square__scene_000000.json
    scene_json_files = sorted(scenes_dir.glob(args.glob_pattern))
    if not scene_json_files:
        # Fallback to original simple pattern if custom yields none
        fallback = sorted(scenes_dir.glob('scene_*.json'))
        if fallback:
            scene_json_files = fallback
            if args.verbose:
                print(f"[info] primary glob pattern '{args.glob_pattern}' matched 0 files; using fallback 'scene_*.json' ({len(scene_json_files)} files)")
    if args.verbose:
        print(f"[info] matched {len(scene_json_files)} scene json files with pattern '{args.glob_pattern}'")
    if args.max_scenes is not None:
        scene_json_files = scene_json_files[:args.max_scenes]

    all_entries: List[Dict[str, Any]] = []
    next_id = 1
    skipped = 0
    skip_reasons = {}
    for idx, sp in enumerate(scene_json_files):
        try:
            entries = build_entries_for_scene(sp, images_dir, next_id, rng, verbose=args.verbose)
        except Exception as e:
            skipped += 1
            reason = str(e)
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if args.verbose:
                print(f"[warn] Skipping {sp.name}: {e}", file=sys.stderr)
            continue
        all_entries.extend(entries)
        next_id = all_entries[-1]['id'] + 1

    with out_file.open('w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    total_scenes = len(scene_json_files)
    produced = len(all_entries)
    print(f"[ok] Wrote {produced} lines (scenes processed={produced//2}, skipped={skipped}) to {out_file}")
    if skipped and not args.verbose:
        # Provide brief aggregated skip reasons
        top = sorted(skip_reasons.items(), key=lambda x: -x[1])[:5]
        print("[info] Top skip reasons (use --verbose for details):")
        for r, c in top:
            print(f"  - {r} (count={c})")


if __name__ == '__main__':
    main()
