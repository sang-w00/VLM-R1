#!/usr/bin/env python3
"""Visualize JSONL action inference results into a single dashboard PNG.

Usage:
  python visualize_jsonl_dashboard.py --input runs/infer/action_1600step_0915/action_accuracy_results.jsonl --output dashboard.png --limit 32

Features:
  * Parse each JSON line
  * Extract <think>...</think> and <answer>...</answer> from model_output
  * Show question, predicted action (answer letter), selected action_letter, correctness
  * Display selected images (up to N per row) next to text
  * Save all entries concatenated vertically into one scrollable-like tall PNG (Matplotlib figure)

Future improvements (TODO):
  - Pagination / multi-page PDF or multi-image output when entries exceed size
  - HTML interactive dashboard
  - Color-coding by error type (wrong action vs wrong verifier answer)
"""
from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

THINK_RE = re.compile(r"<think[^>]*>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>([A-Z])</answer>", re.IGNORECASE)

@dataclass
class QARecord:
    index: int
    scene_index: Optional[int]
    question: str
    think: str
    answer_letter: Optional[str]
    action_letter: Optional[str]
    selected_images: List[str]
    verifier_answer: Optional[str]
    answer_gt: Optional[str]
    correctness: bool
    reward: Optional[float] = None
    raw_model_output: Optional[str] = None


@dataclass
class CombinedRecord:
    """A pairing of primary & baseline model outputs for the same (scene_index, question)."""
    key: Tuple[Optional[int], str]
    primary: QARecord
    baseline: Optional[QARecord]


def extract_think_and_answer(model_output: str) -> Tuple[str, Optional[str]]:
    """Robustly extract think block and answer letter from raw model output.

    Steps:
      1. Try standard <think>...</think> (case-insensitive)
      2. If missing: look for <answer>X</answer>; take preceding reasoning heuristically
      3. Cleanup: strip leading role markers (system/user/assistant) lines
    """
    think = ''
    # Answer (case-insensitive A-L) first
    ans_match = ANSWER_RE.search(model_output)
    answer_letter = ans_match.group(1).upper() if ans_match else None

    # Collect ALL think blocks, choose the LAST non-empty one
    all_thinks = [m.group(1).strip() for m in THINK_RE.finditer(model_output) if m.group(1).strip()]
    if all_thinks:
        think = all_thinks[-1]
    else:
        # Heuristic fallback: take text prior to answer (or full output) after last 'assistant'
        segment = model_output
        if ans_match:
            segment = model_output[:ans_match.start()]
        lower_seg = segment.lower()
        anchor_pos = lower_seg.rfind('assistant')
        if anchor_pos != -1:
            segment = segment[anchor_pos + len('assistant'):]
        segment = segment.strip()[-1500:]  # safety cap
        think = segment
    # Cleanup redundant tags if any remain
    think = re.sub(r"</?think[^>]*>", '', think, flags=re.IGNORECASE).strip()
    return think, answer_letter


def parse_jsonl(path: str, limit: Optional[int] = None, verbose: bool = False) -> List[QARecord]:
    records: List[QARecord] = []
    missing_think = 0  # for statistics only
    with open(path, 'r', encoding='utf-8') as f:
        for line_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {line_i}: {e}")
                continue
            model_output = obj.get('model_output', '')
            think, answer_letter = extract_think_and_answer(model_output)
            if not think:
                missing_think += 1
            action_letter = obj.get('action_letter')
            selected_images = obj.get('selected_images') or []
            verifier_answer = obj.get('verifier_answer')
            answer_gt = obj.get('answer_gt')
            reward = obj.get('reward')
            # Define correctness: reward>0 OR (verifier_answer matches answer_gt) if both present
            correctness = False
            if reward is not None and reward > 0:
                correctness = True
            elif verifier_answer and answer_gt and str(verifier_answer).lower() == str(answer_gt).lower():
                correctness = True
            rec = QARecord(
                index=len(records),
                scene_index=obj.get('scene_index'),
                question=obj.get('question', ''),
                think=think,
                answer_letter=answer_letter,
                action_letter=action_letter,
                selected_images=selected_images,
                verifier_answer=verifier_answer,
                answer_gt=answer_gt,
                correctness=correctness,
                reward=reward,
                raw_model_output=model_output,
            )
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    if verbose:
        print(f"Parsed {len(records)} records. Empty think after extraction: {missing_think}")
    return records


def combine_primary_baseline(primary: List[QARecord], baseline: List[QARecord]) -> List[CombinedRecord]:
    """Align records by (scene_index, question). If no baseline, baseline=None."""
    b_map: Dict[Tuple[Optional[int], str], QARecord] = {}
    for b in baseline:
        key = (b.scene_index, b.question)
        # Keep first occurrence only
        if key not in b_map:
            b_map[key] = b
    combined: List[CombinedRecord] = []
    for p in primary:
        key = (p.scene_index, p.question)
        combined.append(CombinedRecord(key=key, primary=p, baseline=b_map.get(key)))
    return combined


def _load_image(path: str, max_side: int = 256) -> Optional[Image.Image]:
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert('RGB')
        # Resize preserving aspect
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}")
        return None


def _preview(text: str, max_len: int = 280) -> str:
    t = text.replace('\n', ' ').strip()
    return t[:max_len] + ('...' if len(t) > max_len else '')


def _wrap_and_truncate(text: str, width: int = 90, max_lines: int = 8) -> str:
    """Wrap long reasoning text to prevent overflow into adjacent grids.
    width: approximate chars per line (monospace assumption)
    max_lines: limit number of lines; append ellipsis if truncated.
    """
    import textwrap
    cleaned = re.sub(r'\s+', ' ', text.replace('\r', ' ')).strip()
    if not cleaned:
        return ''
    wrapped = textwrap.wrap(cleaned, width=width)
    truncated = False
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        truncated = True
    if truncated and wrapped:
        if not wrapped[-1].endswith('...'):
            wrapped[-1] = (wrapped[-1][: width-3].rstrip('.') + '...')
    return '\n'.join(wrapped)


def render_dashboard(records: List[QARecord], images_per_row: int = 4, add_title: bool = False) -> plt.Figure:
    """Original single-model dashboard rendering."""
    if not records:
        raise ValueError("No records to render")
    base_height_per_record = 2.2
    fig_height = max(4, base_height_per_record * len(records))
    fig = plt.figure(figsize=(18, fig_height))
    outer_gs = GridSpec(len(records), 1, figure=fig, hspace=0.55)
    for i, rec in enumerate(records):
        outer_ax = fig.add_subplot(outer_gs[i])
        outer_ax.axis('off')
        inner_gs = outer_ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.35, 3.05], wspace=0.02)
        text_ax = fig.add_subplot(inner_gs[0])
        img_ax = fig.add_subplot(inner_gs[1])
        text_ax.axis('off')
        img_ax.axis('off')
        status_color = '#2e8b57' if rec.correctness else '#b22222'
        think_block = _wrap_and_truncate(rec.think, width=82, max_lines=7)
        lines = [
            f"[{rec.index}] Scene {rec.scene_index} Q: {rec.question}",
            "Think:",
            think_block if think_block else '(empty)',
            f"Answer Letter: {rec.answer_letter} | Action: {rec.action_letter}",
            f"Verifier: {rec.verifier_answer} | GT: {rec.answer_gt}",
            f"Reward: {rec.reward} | Correct: {rec.correctness}",
        ]
        text_ax.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=10, family='monospace',
                     color=status_color if not rec.correctness else 'black')
        _render_images_into_ax(img_ax, rec.selected_images, images_per_row)
        _outline_axes([text_ax, img_ax], status_color)
    if add_title:
        fig.suptitle('Action Inference Dashboard', fontsize=16)
    return fig


def render_comparison_dashboard(combined: List[CombinedRecord], images_per_row: int = 4,
                                primary_label: str = 'Primary', baseline_label: str = 'Baseline', tight: bool = False,
                                add_title: bool = False) -> plt.Figure:
    """Horizontal comparison: Question + Primary + Baseline (each with its own images)."""
    if not combined:
        raise ValueError('No combined records to render')
    # Increase per-record height & width in non-tight mode for clearer separation & larger images
    # Increase per-record height to allow more reasoning lines visible
    base_height_per_record = 2.5 if tight else 3.5
    fig_height = max(4, base_height_per_record * len(combined))
    fig_width = (26 if images_per_row >= 4 else 22) if tight else (28 if images_per_row >= 4 else 24)
    fig = plt.figure(figsize=(fig_width, fig_height))
    # Increase vertical spacing further in non-tight mode to avoid text overlap between rows
    # Slightly larger vertical spacing between rows so extended think blocks do not collide
    outer_gs = GridSpec(len(combined), 1, figure=fig, hspace=(0.34 if tight else 0.70))
    for i, crecord in enumerate(combined):
        outer_ax = fig.add_subplot(outer_gs[i])
        outer_ax.axis('off')
        # 3 columns: question/meta, primary, baseline
        # Reduced gap between question and primary columns:
        # - Narrower question column, slightly wider primary
        # - Smaller overall wspace (affects both gaps, but primary-baseline still acceptable)
        sub_gs = outer_ax.get_subplotspec().subgridspec(
            1, 3,
            # Widen text panels (primary/baseline) further and slightly shrink the question column.
            # Also slightly reduce inter-panel wspace to bring baseline closer to primary images.
            width_ratios=[0.80, 1.95, 1.90],
            wspace=0.012 if tight else 0.018
        )
        # Question column will itself be split vertically: text (top) + input image (bottom)
        q_container = fig.add_subplot(sub_gs[0])
        q_container.axis('off')
        # Enlarge question image area (increase lower panel share)
        q_sub = q_container.get_subplotspec().subgridspec(2, 1, height_ratios=[0.40, 0.60])
        q_text_ax = fig.add_subplot(q_sub[0])
        q_img_ax = fig.add_subplot(q_sub[1])
        p_ax = fig.add_subplot(sub_gs[1])
        b_ax = fig.add_subplot(sub_gs[2])
        for ax in (q_text_ax, q_img_ax, p_ax, b_ax):
            ax.axis('off')
        p = crecord.primary
        b = crecord.baseline
        # Question + GT meta
        q_lines = [
            f"[{p.index}] Scene {p.scene_index}",
            f"Q: {p.question}",
            f"GT: {p.answer_gt}",
        ]
        q_text_ax.text(0, 1, '\n'.join(q_lines), va='top', ha='left', fontsize=10, weight='bold')
        # Input view image: use first selected image from primary if available
        if p.selected_images:
            first_img = _load_image(p.selected_images[0])
            if first_img is not None:
                q_img_ax.imshow(first_img)
                q_img_ax.set_xticks([]); q_img_ax.set_yticks([])
                q_img_ax.set_title(os.path.basename(p.selected_images[0]), fontsize=6)
            else:
                q_img_ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=7)
        else:
            q_img_ax.text(0.5, 0.5, 'No input image', ha='center', va='center', fontsize=7)
        # Primary block (text top, images bottom)
        _render_model_with_images(fig, p_ax, p, primary_label, images_per_row)
        # Baseline block
        if b is not None:
            _render_model_with_images(fig, b_ax, b, baseline_label, images_per_row)
        else:
            b_ax.text(0.5, 0.5, 'Baseline missing', ha='center', va='center', fontsize=9, color='orange')
        # Outline colors
        _outline_axes([p_ax], '#2e8b57' if p.correctness else '#b22222')
        if b is not None:
            _outline_axes([b_ax], '#2e8b57' if b.correctness else '#b22222')
    if add_title:
        fig.suptitle('Action Inference Comparison Dashboard', fontsize=16 if tight else 18, y=0.995)
        if tight:
            plt.subplots_adjust(top=0.985, left=0.01, right=0.995)
    return fig


def _render_model_block(ax, rec: QARecord, label: str, wrap_width: int = 70, max_lines: int = 6):
    think_wrapped = _wrap_and_truncate(rec.think, width=wrap_width, max_lines=max_lines)
    lines = [
        f"{label}:",
        "  Think:",
        '    ' + think_wrapped.replace('\n', '\n    ') if think_wrapped else '    (empty)',
        f"  Answer Letter: {rec.answer_letter} | Action: {rec.action_letter}",
        f"  Verifier: {rec.verifier_answer} | Reward: {rec.reward} | Correct: {rec.correctness}",
    ]
    color = 'black' if rec.correctness else '#b22222'
    ax.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=9, family='monospace', color=color)


def _render_model_with_images(fig, ax, rec: QARecord, label: str, images_per_row: int):
    # Updated layout: slimmer spacers, wider text, modestly wide images.
    # 5-column subgrid: left spacer | text | mid spacer | images | right spacer
    # Goal: widen reasoning text block & reduce visual gap to following panel.
    sub = ax.get_subplotspec().subgridspec(
        1, 5,
        width_ratios=[0.03, 1.40, 0.18, 1.80, 0.19],
        wspace=0.035
    )
    left_sp = fig.add_subplot(sub[0])
    text_ax = fig.add_subplot(sub[1])
    mid_sp = fig.add_subplot(sub[2])
    img_ax = fig.add_subplot(sub[3])
    right_sp = fig.add_subplot(sub[4])
    for _a in (left_sp, mid_sp, right_sp, text_ax, img_ax):
        _a.axis('off')
    # Increase wrap width due to expanded text area.
    _render_model_block(text_ax, rec, label, wrap_width=44, max_lines=14)
    _render_images_into_ax(img_ax, rec.selected_images, images_per_row)


def _render_images_into_ax(ax, image_paths: List[str], images_per_row: int):
    # Use a dedicated subgrid layout (rows x cols) inside this region to avoid any overlap with text.
    if not image_paths:
        ax.text(0.5, 0.5, 'No images', ha='center', va='center')
        return
    cols = max(1, images_per_row)
    imgs = [(_load_image(p), p) for p in image_paths]
    valid = [pair for pair in imgs if pair[0] is not None]
    if not valid:
        ax.text(0.5, 0.5, 'Images not found', ha='center', va='center')
        return
    n = len(valid)
    rows = (n + cols - 1) // cols
    # Create a subgridspec inside this axis' slot
    sub = ax.get_subplotspec().subgridspec(rows, cols, wspace=0.05, hspace=0.05)
    for idx, (im, path) in enumerate(valid):
        r = idx // cols
        c = idx % cols
        cell_ax = ax.figure.add_subplot(sub[r, c])
        cell_ax.imshow(im)
        cell_ax.set_xticks([])
        cell_ax.set_yticks([])
        cell_ax.set_title(os.path.basename(path), fontsize=6)
    # Hide the placeholder parent axis area
    ax.axis('off')


def _outline_axes(axes, color: str):
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to jsonl file')
    parser.add_argument('--output', required=True, help='Output PNG path')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records')
    parser.add_argument('--images-per-row', type=int, default=4)
    parser.add_argument('--baseline-input', type=str, default=None, help='Optional baseline jsonl to compare')
    parser.add_argument('--primary-label', type=str, default='Primary', help='Label for primary model')
    parser.add_argument('--baseline-label', type=str, default='Baseline', help='Label for baseline model')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--tight', action='store_true', help='Use tighter layout (reduced spacing)')
    parser.add_argument('--primary-wins-only', action='store_true',
                        help='(Comparison mode) Show only cases where primary model is correct and baseline is incorrect')
    parser.add_argument('--show-title', action='store_true', help='Optionally show a figure title (off by default)')
    parser.add_argument('--both-wrong-only', action='store_true',
                        help='(Comparison mode) Show only cases where BOTH primary and baseline are incorrect')
    parser.add_argument('--both-correct-only', action='store_true',
                        help='(Comparison mode) Show only cases where BOTH primary and baseline are correct')
    args = parser.parse_args()

    records = parse_jsonl(args.input, limit=args.limit, verbose=args.verbose)
    if not records:
        print('No records parsed from primary input. Abort.')
        return
    if args.baseline_input:
        baseline_records = parse_jsonl(args.baseline_input, limit=args.limit, verbose=args.verbose)
        combined = combine_primary_baseline(records, baseline_records)
        # Enforce mutual exclusivity of filtering flags
        active_filters = [f for f, on in [
            ('primary_wins_only', args.primary_wins_only),
            ('both_wrong_only', args.both_wrong_only),
            ('both_correct_only', args.both_correct_only)
        ] if on]
        if len(active_filters) > 1:
            print(f"[ERROR] Multiple comparison filters specified: {active_filters}. Choose only one.")
            return
        before = len(combined)
        if args.primary_wins_only:
            combined = [cr for cr in combined if cr.primary.correctness and (cr.baseline is not None and not cr.baseline.correctness)]
            tag = 'primary-wins-only'
        elif args.both_wrong_only:
            combined = [cr for cr in combined if (cr.baseline is not None and (not cr.primary.correctness) and (not cr.baseline.correctness))]
            tag = 'both-wrong-only'
        elif args.both_correct_only:
            combined = [cr for cr in combined if (cr.baseline is not None and cr.primary.correctness and cr.baseline.correctness)]
            tag = 'both-correct-only'
        else:
            tag = None
        if tag is not None:
            if args.verbose:
                print(f"Filtered {tag}: {len(combined)} / {before} remain")
            if not combined:
                print(f'No records satisfy {tag} condition. Abort.')
                return
        fig = render_comparison_dashboard(
            combined,
            images_per_row=args.images_per_row,
            primary_label=args.primary_label,
            baseline_label=args.baseline_label,
            tight=args.tight,
            add_title=args.show_title,
        )
    else:
        fig = render_dashboard(records, images_per_row=args.images_per_row, add_title=args.show_title)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved dashboard to {args.output} (records={len(records)})')


if __name__ == '__main__':
    main()
