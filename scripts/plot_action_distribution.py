#!/usr/bin/env python3
"""Plot distribution of action letters (A-L) from an action_accuracy_results.jsonl file.

Usage:
  python scripts/plot_action_distribution.py \
      --jsonl runs/infer/action_1600step_0915/action_accuracy_results.jsonl \
      --out-img runs/infer/action_1600step_0915/action_letter_distribution.png

Options:
  --jsonl PATH            Input JSONL file (one JSON object per line)
  --out-img PATH          Output image path (PNG, PDF, etc.)
  --title TEXT            Custom plot title
  --show                  Display the figure in a window (if environment supports GUI)
  --save-csv PATH         Also save raw counts and percentages to a CSV file
  --normalize             Show percentage values on y-axis instead of raw counts
  --cumulative            Additionally plot cumulative percentage line
  --min-count N           Filter out letters with raw count < N (default 0)
  --dpi NUM               Figure DPI (default 120)

Example:
  python scripts/plot_action_distribution.py \
      --jsonl runs/infer/action_1600step_0915/action_accuracy_results.jsonl \
      --out-img runs/infer/action_1600step_0915/action_letter_distribution.png \
      --normalize --cumulative

If multiple files:
  python scripts/plot_action_distribution.py --jsonl file1.jsonl file2.jsonl --out-img merged.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter, OrderedDict
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")  # Safe for headless environments; overridden if --show
import matplotlib.pyplot as plt

VALID_LETTERS = [chr(ord('A') + i) for i in range(12)]  # A-L


def parse_args():
    p = argparse.ArgumentParser(description="Plot distribution of action letters A-L from JSONL logs.")
    p.add_argument('--jsonl', nargs='+', required=True, help='Input JSONL file(s)')
    p.add_argument('--out-img', required=True, help='Output image path (e.g., distribution.png)')
    p.add_argument('--title', default=None, help='Custom title (default auto)')
    p.add_argument('--show', action='store_true', help='Show the plot interactively')
    p.add_argument('--save-csv', default=None, help='Optional CSV export path')
    p.add_argument('--normalize', action='store_true', help='Plot percentages instead of raw counts')
    p.add_argument('--cumulative', action='store_true', help='Add cumulative percentage line')
    p.add_argument('--min-count', type=int, default=0, help='Filter out letters with raw count < N')
    p.add_argument('--dpi', type=int, default=120)
    return p.parse_args()


def load_action_letters(files: List[str]) -> Counter:
    counter = Counter()
    total_lines = 0
    bad_lines = 0
    for fp in files:
        path = Path(fp)
        if not path.exists():
            print(f"[warn] File not found: {fp}", file=sys.stderr)
            continue
        with path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                letter = obj.get('action_letter') or obj.get('action') or ''
                if not isinstance(letter, str):
                    continue
                letter = letter.strip().upper()
                if letter in VALID_LETTERS:
                    counter[letter] += 1
    if bad_lines:
        print(f"[info] Skipped {bad_lines} malformed JSON lines out of {total_lines}")
    return counter


def ordered_counts(counter: Counter, min_count: int) -> OrderedDict:
    od = OrderedDict()
    for letter in VALID_LETTERS:
        c = counter.get(letter, 0)
        if c >= min_count:
            od[letter] = c
    return od


def save_csv(path: str, counts: OrderedDict, normalize: bool):
    import csv
    total = sum(counts.values()) or 1
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['letter', 'count', 'percentage'])
        for letter, c in counts.items():
            w.writerow([letter, c, f"{c/total*100:.4f}"])
    print(f"[ok] Saved CSV: {path}")


def plot_distribution(counts: OrderedDict, out_img: str, title: str | None, normalize: bool, cumulative: bool, dpi: int, show: bool):
    letters = list(counts.keys())
    raw_counts = list(counts.values())
    total = sum(raw_counts) or 1
    if normalize:
        values = [c / total * 100 for c in raw_counts]
        ylabel = 'Percentage (%)'
    else:
        values = raw_counts
        ylabel = 'Count'

    plt.figure(figsize=(max(6, len(letters) * 0.6), 4))
    bars = plt.bar(letters, values, color='#3477eb', edgecolor='black', alpha=0.85)

    # Annotate bars
    for b, v, rc in zip(bars, values, raw_counts):
        if normalize:
            txt = f"{v:.1f}%\n({rc})"
        else:
            txt = str(rc)
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), txt, ha='center', va='bottom', fontsize=9)

    if cumulative:
        cum_vals = []
        s = 0
        for rc in raw_counts:
            s += rc
            cum_vals.append(s / total * 100)
        ax2 = plt.twinx()
        ax2.plot(letters, cum_vals, color='orange', marker='o', linewidth=1.5, label='Cumulative %')
        ax2.set_ylabel('Cumulative (%)')
        ax2.set_ylim(0, 105)
        ax2.grid(False)
        ax2.legend(loc='upper left')

    plt.title(title or f"Action Letter Distribution (n={total})")
    plt.xlabel('Action Letter')
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    out_path = Path(out_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    print(f"[ok] Saved figure: {out_path}")
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[warn] Could not show figure: {e}")
    plt.close()


def main():
    args = parse_args()
    counter = load_action_letters(args.jsonl)
    counts = ordered_counts(counter, args.min_count)
    if not counts:
        print('[error] No valid action letters found (A-L). Exiting.')
        sys.exit(1)
    # Print summary
    total = sum(counts.values())
    print('Letter\tCount\t%')
    for l, c in counts.items():
        print(f"{l}\t{c}\t{c/total*100:.2f}")
    print(f"Total\t{total}")

    if args.save_csv:
        save_csv(args.save_csv, counts, args.normalize)

    plot_distribution(counts, args.out_img, args.title, args.normalize, args.cumulative, args.dpi, args.show)


if __name__ == '__main__':
    main()
