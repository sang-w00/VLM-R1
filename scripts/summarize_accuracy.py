import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple, DefaultDict
from collections import defaultdict


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append(obj)
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return entries


def compute_accuracy(entries: List[Dict[str, Any]]) -> float:
    if not entries:
        return 0.0
    total = 0
    correct = 0.0
    for e in entries:
        if 'reward' not in e:
            continue
        total += 1
        try:
            r = float(e['reward'])
        except Exception:
            r = 0.0
        # Treat reward >= 0.5 as correct by default; adjust if needed
        correct += 1.0 if r >= 0.5 else 0.0
    return (correct / total) if total > 0 else 0.0


def _resolve_group_key(e: Dict[str, Any], preferred: Optional[str] = None) -> Optional[Any]:
        """Choose a grouping key for circular evaluation.

        Supports comma-separated multi-key (e.g., 'scene_index,question') and falls
        back to single-key auto-detection when none provided or unavailable.

        Priority when preferred is provided:
            - If preferred contains commas, return a tuple of values in that order.
                If all are missing (None), fall back to defaults.
            - If preferred is a single key and present, return its value; otherwise fallback.

        Fallback order:
            1) 'circular_group'
            2) 'scene_index'
            3) 'id'
        Returns None if nothing is available.
        """
        if preferred:
                keys = [k.strip() for k in str(preferred).split(',') if k.strip()]
                if len(keys) > 1:
                        vals = tuple(e.get(k) for k in keys)
                        # Use tuple if at least one key exists in entry
                        if any(k in e for k in keys):
                                return vals
                elif len(keys) == 1:
                        k = keys[0]
                        if k in e:
                                return e.get(k)
        for k in ('circular_group', 'scene_index', 'id'):
                if k in e:
                        return e.get(k)
        return None


def compute_circular_accuracy(
    entries: List[Dict[str, Any]],
    threshold: float = 0.5,
    group_key: Optional[str] = None,
) -> Tuple[float, int, int]:
    """Compute circular accuracy by grouping variants per question and requiring
    all variants in a group to be correct (reward >= threshold).

    Returns (accuracy, total_groups, correct_groups).
    Groups with no valid 'reward' entries are ignored.
    """
    if not entries:
        return 0.0, 0, 0

    groups: DefaultDict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        # only consider entries that actually have a reward field
        if 'reward' not in e:
            continue
        gkey = _resolve_group_key(e, preferred=group_key)
        if gkey is None:
            # If no usable group key, treat each entry as its own group using index fallback
            gkey = id(e)
        groups[gkey].append(e)

    total_groups = 0
    correct_groups = 0
    for gkey, items in groups.items():
        if not items:
            continue
        total_groups += 1
        group_correct = True
        for e in items:
            try:
                r = float(e.get('reward', 0.0))
            except Exception:
                r = 0.0
            if r < threshold:
                group_correct = False
                break
        if group_correct:
            correct_groups += 1

    acc = (correct_groups / total_groups) if total_groups > 0 else 0.0
    return acc, total_groups, correct_groups


def main():
    p = argparse.ArgumentParser(description='Summarize accuracy from JSONL results produced by test_action_accuracy.py or training verifier outputs.')
    p.add_argument('--files', type=str, nargs='+', required=True, help='One or more JSONL files to aggregate')
    p.add_argument('--threshold', type=float, default=0.5, help='Threshold to count as correct (default: 0.5)')
    p.add_argument('--circular', action='store_true', help='Enable circular evaluation: group variants per question and require all to be correct')
    p.add_argument('--group-key', type=str, default=None, help="Group key for circular evaluation (default in circular mode: scene_index,question). You can also pass multiple keys separated by commas, e.g., 'scene_index,question'. If not set and not in circular mode, sample-wise accuracy is reported.")
    args = p.parse_args()

    all_entries: List[Dict[str, Any]] = []
    for path in args.files:
        if not os.path.exists(path):
            print(f"[warn] file not found: {path}")
            continue
        entries = load_jsonl(path)
        all_entries.extend(entries)

    if not all_entries:
        print('No entries found.')
        return

    if args.circular:
        # Default grouping for circular evaluation is by (scene_index, question)
        effective_group_key = args.group_key or 'scene_index,question'
        acc, total_groups, correct_groups = compute_circular_accuracy(
            all_entries, threshold=args.threshold, group_key=effective_group_key
        )
        print(f"Groups: {total_groups}")
        print(f"Correct Groups: {correct_groups}")
        print(f"Circular Accuracy: {acc:.4f}")
    else:
        # Apply threshold during counting
        total = 0
        correct = 0
        for e in all_entries:
            if 'reward' not in e:
                continue
            total += 1
            try:
                r = float(e['reward'])
            except Exception:
                r = 0.0
            if r >= args.threshold:
                correct += 1

        acc = (correct / total) if total > 0 else 0.0
        print(f"Samples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()
