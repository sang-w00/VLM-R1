#!/usr/bin/env python3
"""Attach ground-truth action choice (A or B) to an existing numbered_square training JSONL.

Logic:
  For each line in the training JSONL (already produced by build_numbered_square_training_jsonl.py),
  we parse:
    - id
    - image path (to extract scene index)
    - human prompt (conversations[0].value)
      * Determine which side question asks: 'left side' or 'right side'
      * Extract color mapping for Actions A / B
  Then we load the corresponding scene JSON (has two small cubes). Using their 3d_coords[0] (x):
      left_cube  = cube with smaller x
      right_cube = cube with larger x
  Ground-truth action:
      If question asks left  -> action that corresponds to left_cube color
      If question asks right -> action that corresponds to right_cube color

Outputs:
  (default) A new JSONL where each original sample dict gains a key 'gt_action': 'A' or 'B'.
  (optional) If --actions-only is passed, output a compact JSONL with only {"id": ..., "gt_action": ...}.

Usage:
  python dataset/attach_gt_actions.py \
    --training-jsonl dataset/numbered_square_training.jsonl \
    --scenes-dir /home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/numbered_square/training/scenes \
    --out-jsonl dataset/numbered_square_training_with_actions.jsonl

Only mapping file:
  python dataset/attach_gt_actions.py \
    --training-jsonl dataset/numbered_square_training.jsonl \
    --scenes-dir /home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/numbered_square/training/scenes \
    --out-jsonl dataset/numbered_square_training_actions_only.jsonl \
    --actions-only
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Any, Tuple

SCENE_RE = re.compile(r'scene_(\d{6})')
A_B_COLORS_RE = re.compile(
    r'Actions:\s*A\.[^\n]*?from the ([a-zA-Z]+) box\'s perspective\s*B\.[^\n]*?from the ([a-zA-Z]+) box\'s perspective',
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Attach GT action (A/B) to training JSONL")
    ap.add_argument('--training-jsonl', required=True, help='Existing training JSONL (with prompts)')
    ap.add_argument('--scenes-dir', required=True, help='Directory containing scene JSON files')
    ap.add_argument('--glob-pattern', default='*scene_*.json', help='Scene filename pattern (default *scene_*.json)')
    ap.add_argument('--out-jsonl', required=True, help='Output JSONL path to write')
    ap.add_argument('--actions-only', action='store_true', help='Write only id->gt_action minimal lines')
    ap.add_argument('--strict', action='store_true', help='Fail on mismatch instead of warning & skip')
    return ap.parse_args()


def load_scene(scene_path: Path) -> Dict[str, Any]:
    return json.loads(scene_path.read_text(encoding='utf-8'))


def find_scene_file(scenes_dir: Path, pattern: str, scene_index: int) -> Path | None:
    # First try common simple name
    simple = scenes_dir / f'scene_{scene_index:06d}.json'
    if simple.exists():
        return simple
    # Fallback: search by provided glob pattern and matching scene_index substring
    for p in scenes_dir.glob(pattern):
        if f'scene_{scene_index:06d}' in p.name:
            return p
    return None


def extract_scene_index(image_path: str) -> int | None:
    m = SCENE_RE.search(image_path)
    if not m:
        return None
    return int(m.group(1))


def extract_colors_from_prompt(prompt: str) -> Tuple[str, str] | None:
    m = A_B_COLORS_RE.search(prompt)
    if not m:
        return None
    a_color, b_color = m.group(1).lower(), m.group(2).lower()
    return a_color, b_color


def extract_side_from_prompt(prompt: str) -> str | None:
    low = prompt.lower()
    if ' left side of the gray cube' in low:
        return 'left'
    if ' right side of the gray cube' in low:
        return 'right'
    return None


def derive_left_right_colors(scene: Dict[str, Any]) -> Tuple[str, str]:
    small_cubes = [o for o in scene.get('objects', []) if o.get('shape') == 'cube' and o.get('size') == 'small']
    if len(small_cubes) != 2:
        raise ValueError(f"Expected 2 small cubes, got {len(small_cubes)}")
    # Sort by x (3d_coords[0]) ascending
    sorted_cubes = sorted(small_cubes, key=lambda o: float(o['3d_coords'][0]))
    left_color = sorted_cubes[0]['color'].lower().strip()
    right_color = sorted_cubes[1]['color'].lower().strip()
    return left_color, right_color


def main():
    args = parse_args()
    training_path = Path(args.training_jsonl)
    scenes_dir = Path(args.scenes_dir)
    out_path = Path(args.out_jsonl)

    if not training_path.exists():
        print(f"[error] training jsonl not found: {training_path}", file=sys.stderr); sys.exit(1)
    if not scenes_dir.exists():
        print(f"[error] scenes dir not found: {scenes_dir}", file=sys.stderr); sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    wrote = 0
    skipped = 0

    with training_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] line {line_no}: invalid JSON, skipping", file=sys.stderr)
                skipped += 1
                continue
            image_path = obj.get('image') or ''
            scene_index = extract_scene_index(image_path)
            if scene_index is None:
                msg = f"no scene index in image path '{image_path}'"
                if args.strict: raise RuntimeError(msg)
                print(f"[warn] line {line_no}: {msg}", file=sys.stderr); skipped +=1; continue

            prompt = ''
            try:
                prompt = obj['conversations'][0]['value']
            except Exception:
                msg = 'missing conversations[0].value'
                if args.strict: raise RuntimeError(msg)
                print(f"[warn] line {line_no}: {msg}", file=sys.stderr); skipped +=1; continue

            side = extract_side_from_prompt(prompt)
            if side is None:
                msg = 'cannot detect side (left/right)'
                if args.strict: raise RuntimeError(msg)
                print(f"[warn] line {line_no}: {msg}", file=sys.stderr); skipped +=1; continue

            colors_map = extract_colors_from_prompt(prompt)
            if colors_map is None:
                msg = 'cannot parse colors for A/B'
                if args.strict: raise RuntimeError(msg)
                print(f"[warn] line {line_no}: {msg}", file=sys.stderr); skipped +=1; continue
            a_color, b_color = colors_map

            scene_file = find_scene_file(scenes_dir, args.glob_pattern, scene_index)
            if scene_file is None or not scene_file.exists():
                msg = f"scene file not found for index {scene_index}"
                if args.strict: raise RuntimeError(msg)
                print(f"[warn] line {line_no}: {msg}", file=sys.stderr); skipped +=1; continue

            try:
                scene = load_scene(scene_file)
                left_color, right_color = derive_left_right_colors(scene)
            except Exception as e:
                if args.strict: raise
                print(f"[warn] line {line_no}: scene parse error: {e}", file=sys.stderr); skipped +=1; continue

            target_color = left_color if side == 'left' else right_color

            # Map to action
            if target_color == a_color:
                gt_action = 'A'
            elif target_color == b_color:
                gt_action = 'B'
            else:
                # Try swapping (maybe order randomized but colors mismatch due to casing / synonyms)
                if target_color.lower() == a_color.lower():
                    gt_action = 'A'
                elif target_color.lower() == b_color.lower():
                    gt_action = 'B'
                else:
                    msg = f"target_color '{target_color}' not found in A/B mapping (A={a_color}, B={b_color})"
                    if args.strict: raise RuntimeError(msg)
                    print(f"[warn] line {line_no}: {msg}", file=sys.stderr)
                    skipped +=1
                    continue

            if args.actions_only:
                out_obj = { 'id': obj.get('id'), 'gt_action': gt_action }
            else:
                obj['gt_action'] = gt_action
                out_obj = obj

            fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            wrote += 1

    print(f"[ok] processed={total} wrote={wrote} skipped={skipped} output={out_path}")

    if skipped:
        print("[info] Use --strict to fail fast on warnings, or inspect the warnings above.")

if __name__ == '__main__':
    main()
