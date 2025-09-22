import os
import json
import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq

# We reuse logic from grpo_jsonl without importing the whole training script to avoid side effects
# If project structure changes, consider refactoring shared utilities into a separate module.

# NOTE: Action space updated to align with grpo_jsonl A/B setup.
# Historically this script supported A-K plus L (no rotation). The GRPO code
# reduced actions to only A/B (mapping to view_01 / view_02) and treats any
# other output as 'no extra view'. We keep 'L' as a fallback representing
# "no additional view" for backward compatibility with downstream logic that
# expects a letter, but the generation & extraction now only allow A or B.
ACTION_LETTERS = {
    'A': 1,  # view_01.png
    'B': 2,  # view_02.png
    'L': 0,  # no extra view / invalid fallback
}

ALIAS_MAP = {
    "qwen2.5vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5_vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5_vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}

def resolve_model_path(raw: str) -> str:
    return ALIAS_MAP.get(raw.strip().lower(), raw)


def swap_gt_action_letter(letter: Optional[str]) -> Optional[str]:
    """Swap A <-> B for ground-truth action; keep others as-is.

    Examples:
      'A' -> 'B'
      'B' -> 'A'
      'L' or None -> same (used only as fallback)
    """
    if not isinstance(letter, str):
        return letter
    l = letter.strip().upper()
    if l == 'A':
        return 'B'
    if l == 'B':
        return 'A'
    return letter


def swap_choices_in_prompt(prompt: str) -> str:
    """Swap the textual descriptions of choices A and B inside the Actions section.

    This targets prompts like:
      "Actions: A. <descA> B. <descB> The question is as follow: ..."

    We preserve everything outside the Actions block. If the regex fails, we fall
    back to a conservative heuristic swapping by labels.
    """
    if not isinstance(prompt, str):
        return prompt
    text = prompt
    # Try to isolate Actions block up to the next "The question is" or end of string
    m = re.search(r"(Actions:\s*)(A\.)\s*(.*?)(\s*)(B\.)\s*(.*?)(?=(The question is|$))",
                  text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        prefix_actions = m.group(1)
        a_label = m.group(2)  # 'A.'
        a_desc = m.group(3)
        spacer = m.group(4)   # whitespace between A and B parts
        b_label = m.group(5)  # 'B.'
        b_desc = m.group(6)
        tail_anchor = text[m.end():]  # from anchor to end
        swapped_block = f"{prefix_actions}{a_label} {b_desc} {spacer}{b_label} {a_desc}"
        return text[:m.start()] + swapped_block + tail_anchor

    # Fallback: swap lines starting with "A." and "B." after 'Actions:'
    parts = re.split(r"(Actions:)", text, flags=re.IGNORECASE)
    if len(parts) >= 3:
        head = ''.join(parts[:-1])  # everything up to the last split
        after_actions = parts[-1]
        # Find first occurrences of A. and B. in the after_actions segment
        ma = re.search(r"A\.\s*([\s\S]*?)\s*(?=B\.|$)", after_actions)
        mb = re.search(r"B\.\s*([\s\S]*?)(?=(The question is|$))", after_actions, flags=re.IGNORECASE)
        if ma and mb:
            a_desc = ma.group(1)
            b_desc = mb.group(1)
            # Reconstruct conservatively
            before_a = after_actions[:ma.start()]
            after_b_end = mb.end()
            tail = after_actions[after_b_end:]
            swapped = f"{before_a}A. {b_desc} B. {a_desc}{tail}"
            return head + swapped
    # If nothing matched, return original prompt unchanged
    return text

def extract_final_answer_letter(text: str) -> str:
    """Extract the final action letter (A or B) from the model output.

    Updated to mirror the newer GRPO (grpo_jsonl) logic which restricts the
    action space to only two choices (A/B). Any deviation (missing <answer>
    tag, multi-character string, letter outside A/B) results in fallback 'L'
    meaning "no valid extra view".

    Parsing rules:
      - Find all <answer>...</answer> blocks (case-insensitive); take the last.
      - Trim whitespace; allow a single trailing period.
      - Uppercase; if exactly 'A' or 'B' -> return it.
      - Else return 'L'.
    """
    blocks = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return 'L'
    candidate_raw = blocks[-1].strip()
    # Allow one trailing period (e.g., "A.")
    if candidate_raw.endswith('.') and candidate_raw.count('.') == 1:
        candidate_core = candidate_raw[:-1].strip()
    else:
        candidate_core = candidate_raw
    letter = candidate_core.upper()
    return letter if letter in ('A', 'B') and len(letter) == 1 else 'L'

def build_additional_view(primary_path: str, letter: str, mapping: Optional[Dict[str, int]] = None) -> Optional[str]:
    """Return additional view path based on action letter and mapping.

    mapping: optional override for {'A':1, 'B':2, 'L':0} to support dynamic
             left/right-dependent behavior.
    """
    mapping = mapping or ACTION_LETTERS
    if letter not in mapping or letter == 'L':
        return None
    idx = mapping[letter]
    base_dir = os.path.dirname(primary_path)
    candidate = os.path.join(base_dir, f"view_{idx:02d}.png")
    return candidate if os.path.exists(candidate) else None

def _question_side(question: str) -> Optional[str]:
    """Detect whether the question asks about left or right.

    Returns 'left', 'right', or None if unclear. Supports English and Korean keywords.
    """
    if not question:
        return None
    q = question.lower()
    # English
    if re.search(r"\bleft\b", q):
        return 'left'
    if re.search(r"\bright\b", q):
        return 'right'
    # Korean (simple heuristics)
    if any(tok in q for tok in ['왼쪽', '좌측', '왼 편', '왼', '좌']):
        return 'left'
    if any(tok in q for tok in ['오른쪽', '우측', '오른 편', '오른', '우']):
        return 'right'
    return None

def default_accuracy_reward(pred: str, gt: str) -> float:
    # Mirror simplified logic: extract <answer> blocks; fallback to raw
    pred_blocks = re.findall(r'<answer>(.*?)</answer>', pred, re.DOTALL | re.IGNORECASE)
    gt_blocks = re.findall(r'<answer>(.*?)</answer>', gt, re.DOTALL | re.IGNORECASE)
    pred_ans = pred_blocks[-1].strip() if pred_blocks else pred.strip()
    gt_ans = gt_blocks[-1].strip() if gt_blocks else gt.strip()
    # simple normalization
    def norm(x):
        return x.strip().rstrip('.').lower()
    return 1.0 if norm(pred_ans) == norm(gt_ans) else 0.0

_verifier_model = None
_verifier_processor = None

def initialize_verifier(verifier_model_path: str):
    global _verifier_model, _verifier_processor
    if _verifier_model is not None:
        return _verifier_model, _verifier_processor
    resolved = resolve_model_path(verifier_model_path)
    from transformers import AutoProcessor, AutoModelForVision2Seq
    _verifier_processor = AutoProcessor.from_pretrained(resolved, trust_remote_code=True)
    _verifier_model = AutoModelForVision2Seq.from_pretrained(
        resolved,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    _verifier_model.eval()
    for p in _verifier_model.parameters():
        p.requires_grad_(False)
    return _verifier_model, _verifier_processor

@torch.no_grad()
def verifier_answer(image_paths: List[str], question: str, verifier_model_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Run verifier model and return (final_answer, raw_generation).

    final_answer: cleaned short answer used for reward
    raw_generation: entire decoded string (for analysis)
    """
    model, processor = initialize_verifier(verifier_model_path)
    try:
        pil_images = [Image.open(p).convert('RGB') for p in image_paths]
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": f"Answer the question concisely. Question: {question}"}
            ]
        }]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=pil_images, return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        raw_out = processor.batch_decode(gen, skip_special_tokens=True)[0]
        blocks = re.findall(r'<answer>(.*?)</answer>', raw_out, re.DOTALL | re.IGNORECASE)
        if blocks:
            ans = blocks[-1].strip()
        else:
            ans = raw_out.strip().split('\n')[-1].strip()
        ans = ans.rstrip('.').strip()
        return ans, raw_out
    except Exception as e:
        print(f"[test_action_accuracy] verifier failed: {e}")
        return None, None

@torch.no_grad()
def generate_model_answer(model, processor, image_paths: List[str], question: str, max_new_tokens: int = 128) -> str:
    """Generate a model answer constrained to A/B actions.

    Instruction aligns with updated two-action policy:
      1. Produce internal reasoning inside <think>...</think>.
      2. Produce final action inside <answer>...</answer> with EXACTLY one
         uppercase letter: A or B.
      3. No trailing punctuation or extra text after </answer>.
      4. If uncertain, pick the best of A or B.
    """
    pil_images = [Image.open(p).convert('RGB') for p in image_paths]
    instruction = (
        f"{question} First output the thinking process in <think> </think> tags and then output the final answer "
        f"in <answer> </answer> tags. The text between <answer> and </answer> must be exactly one uppercase letter "
        f"A or B. No spaces, words, punctuation, or additional characters are allowed. If uncertain, choose the best "
        f"single letter (A or B). Do not output anything after </answer>."
    )
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in pil_images],
            {"type": "text", "text": instruction}
        ]
    }]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=pil_images, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return out

def load_test_questions(path: str) -> List[Dict[str, Any]]:
    """Load questions supporting two formats:
    1. Original JSON file (list or {"questions": [...]})
    2. JSONL file where each line has: {id, image, conversations:[{"from":"human","value":prompt_with_question}, {"from":"gpt","value":answer}]}

    Returns list of dicts normalized to keys:
        scene_index: int
        question: str
        answer: str
        image: str (primary image path)
    """
    # Heuristic: if file extension is .jsonl treat as JSONL lines
    entries: List[Dict[str, Any]] = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                image_path = obj.get('image')
                convs = obj.get('conversations', [])
                # Extract question: assume first human message contains question after last '<image>' marker or after 'The question is as follow:' pattern
                question_text = ''
                raw_prompt = None
                if convs:
                    human_msg = next((c for c in convs if c.get('from') == 'human'), None)
                    if human_msg:
                        raw_val = human_msg.get('value', '')
                        raw_prompt = raw_val  # keep full original prompt for action model
                        # Look for last occurrence of '<image>' then text after it
                        if '<image>' in raw_val:
                            after = raw_val.split('<image>')[-1]
                            question_text = after.strip().rstrip('?').rstrip('.') + '?' if after.strip() else raw_val[-200:]
                        else:
                            question_text = raw_val.strip()
                        # If embedded label like "The question is as follow:" remove leading phrase
                        m = re.search(r'The question is as follow:?\s*(.*)', question_text, re.IGNORECASE | re.DOTALL)
                        if m:
                            question_text = m.group(1).strip()
                answer_text = ''
                if convs:
                    gpt_msg = next((c for c in convs if c.get('from') == 'gpt'), None)
                    if gpt_msg:
                        answer_text = gpt_msg.get('value', '').strip()
                # Derive scene index from path pattern .../scene_XXXXXX/view_00.png
                scene_index = obj.get('id')
                if image_path:
                    mscene = re.search(r'scene_(\d+)', image_path)
                    if mscene:
                        scene_index = int(mscene.group(1))
                # Optional ground-truth action (e.g., 'A' or 'B') for evaluation-only mode
                gt_action = obj.get('gt_action')
                gt_action = gt_action.strip().upper() if isinstance(gt_action, str) else None

                entries.append({
                    'scene_index': scene_index if isinstance(scene_index, int) else int(scene_index) if str(scene_index).isdigit() else len(entries),
                    'question': question_text,
                    'answer': answer_text,
                    'image': image_path,
                    'raw_prompt': raw_prompt,
                    'gt_action': gt_action,
                })
        return entries
    # Fallback original JSON logic
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'questions' in data:
        return data['questions']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unsupported questions file format.")

# scenes_root support removed: images are now expected to be given explicitly in the JSONL input.

@dataclass
class ResultEntry:
    scene_index: int
    question: str
    answer_gt: str
    model_output: str  # full generation from the action-selection (trained) model (includes thinking)
    action_letter: str
    selected_images: List[str]
    verifier_answer: Optional[str]
    verifier_question: Optional[str]
    reward: float
    # Circular-eval metadata
    variant: str = "orig"  # 'orig' or 'swapped'
    circular_group: Optional[int] = None

    def to_dict(self):
        return {
            'scene_index': self.scene_index,
            'question': self.question,
            'answer_gt': self.answer_gt,
            'model_output': self.model_output,
            'action_letter': self.action_letter,
            'selected_images': self.selected_images,
            'verifier_answer': self.verifier_answer,
            'verifier_question': self.verifier_question,
            'reward': self.reward,
            'variant': self.variant,
            'circular_group': self.circular_group,
        }

def run(args):
    # When using GT actions only, we can avoid loading the action model entirely.
    if args.use_gt_action:
        model = None
        processor = None
        print("Using ground-truth actions: skipping action model loading and inference.")
    else:
        model_path = resolve_model_path(args.model_path)
        print(f"Loading model: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Choose loader (vision2seq vs causal) heuristically
        if any(tag in model_path.lower() for tag in ['2.5-vl', '2_5-vl', '2.5_vl', '2_5_vl', 'qwen2.5-vl', 'qwen2.5']):
            ModelCls = AutoModelForVision2Seq
        else:
            # fallback (may adjust if project has specific class)
            ModelCls = AutoModelForVision2Seq
        model = ModelCls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    questions = load_test_questions(args.questions_file)
    total_loaded = len(questions)
    if args.limit is not None and args.limit > 0:
        questions = questions[:args.limit]
        print(f"Loaded {total_loaded} questions (limiting to first {len(questions)})")
    else:
        print(f"Loaded {total_loaded} questions")

    results: List[ResultEntry] = []
    circular_pairs_total = 0
    circular_pairs_correct = 0

    def evaluate_variant(primary_img: str, question_text: str, raw_prompt_text: str, gt_action_letter: Optional[str], answer_gt_text: str, variant_name: str, group_id: Optional[int]) -> Tuple[ResultEntry, float]:
        """Evaluate one variant (original or swapped) and return (ResultEntry, reward)."""
        nonlocal processor, model
        # Decide action letter: model inference or GT action
        if args.use_gt_action:
            action_letter = (gt_action_letter or 'L')
            if action_letter is None:
                action_letter = 'L'
            action_letter = action_letter.strip().upper() if isinstance(action_letter, str) else 'L'
            if action_letter not in ('A', 'B'):
                action_letter = 'L'
            model_output = f"<think></think><answer>{action_letter}</answer>"
            side = _question_side(question_text)
            mapping = ACTION_LETTERS
            if (side == 'left' and action_letter == 'B') or (side == 'right' and action_letter == 'A'):
                mapping = {'A': 2, 'B': 1, 'L': 0}
        else:
            model_output = generate_model_answer(model, processor, [primary_img], raw_prompt_text, max_new_tokens=args.max_new_tokens)
            action_letter = extract_final_answer_letter(model_output)
            side = _question_side(question_text)
            mapping = ACTION_LETTERS
            ga = gt_action_letter.strip().upper() if isinstance(gt_action_letter, str) else None
            if isinstance(ga, str) and ga in ('A', 'B') and side in ('left', 'right'):
                if (side == 'left' and ga == 'B') or (side == 'right' and ga == 'A'):
                    mapping = {'A': 2, 'B': 1, 'L': 0}

        add_view = build_additional_view(primary_img, action_letter, mapping)
        selected_for_verifier = [primary_img] + ([add_view] if add_view else [])
        verifier_question = question_text
        verifier_ans, _ = verifier_answer(selected_for_verifier, verifier_question, args.verifier_model_path)
        if verifier_ans is None:
            reward = 0.0
        else:
            reward = default_accuracy_reward(f"<answer>{verifier_ans}</answer>", f"<answer>{answer_gt_text}</answer>")
        entry = ResultEntry(
            scene_index=int(scene_index),
            question=question_text,
            answer_gt=answer_gt_text,
            model_output=model_output,
            action_letter=action_letter,
            selected_images=selected_for_verifier,
            verifier_answer=verifier_ans,
            verifier_question=verifier_question,
            reward=reward,
            variant=variant_name,
            circular_group=group_id,
        )
        return entry, reward

    for q in questions:
        # Normalized entries from JSONL loader expected (must contain image path)
        scene_index = q.get('scene_index') if isinstance(q, dict) else q['scene_index']
        question_text = q.get('question') if isinstance(q, dict) else q['question']  # extracted concise question for verifier
        gt_answer = q.get('answer') or q.get('answer_gt') or q.get('gt')
        if gt_answer is None:
            gt_answer = ""
        primary_img = q.get('image') if isinstance(q, dict) else q['image']
        raw_prompt = q.get('raw_prompt', question_text)
        if not primary_img:
            print(f"[warn] skip scene {scene_index}: missing image path (provide in JSONL)")
            continue
        if not os.path.exists(primary_img):
            print(f"[warn] image not found for scene {scene_index}: {primary_img}")
            continue

        # Circular evaluation: evaluate original + swapped choices; count as correct only if both are correct.
        if args.circular_eval:
            circular_pairs_total += 1
            gt_action_letter = (q.get('gt_action') if isinstance(q, dict) else None)
            # original variant
            entry_orig, reward_orig = evaluate_variant(primary_img, question_text, raw_prompt, gt_action_letter, gt_answer, 'orig', int(scene_index))
            results.append(entry_orig)
            if args.verbose:
                print(f"[test][orig] scene={scene_index} action={entry_orig.action_letter} reward={reward_orig:.3f} gt_action={gt_action_letter}")
            # swapped variant: swap the Actions text and flip gt_action
            swapped_prompt = swap_choices_in_prompt(raw_prompt or '')
            swapped_gt = swap_gt_action_letter(gt_action_letter)
            entry_swap, reward_swap = evaluate_variant(primary_img, question_text, swapped_prompt, swapped_gt, gt_answer, 'swapped', int(scene_index))
            results.append(entry_swap)
            if args.verbose:
                print(f"[test][swap] scene={scene_index} action={entry_swap.action_letter} reward={reward_swap:.3f} gt_action={swapped_gt}")

            if (reward_orig >= 1.0) and (reward_swap >= 1.0):
                circular_pairs_correct += 1
        else:
            # Legacy single-variant evaluation
            gt_action_for_mapping = q.get('gt_action') if isinstance(q, dict) else None
            entry_single, reward_single = evaluate_variant(primary_img, question_text, raw_prompt, gt_action_for_mapping, gt_answer, 'orig', int(scene_index))
            results.append(entry_single)
            if args.verbose:
                print(f"[test] scene={scene_index} action={entry_single.action_letter} reward={reward_single:.3f} gt_action={gt_action_for_mapping}")

    # Save
    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(results)} results to {args.output_file}")

    # Final accuracy
    if args.circular_eval:
        if circular_pairs_total > 0:
            print(f"Circular accuracy: {circular_pairs_correct / circular_pairs_total:.4f} ({circular_pairs_correct}/{circular_pairs_total})")
        else:
            print("Circular accuracy: N/A (no pairs evaluated)")
    else:
        if results:
            mean_acc = sum(r.reward for r in results) / len(results)
            print(f"Final accuracy: {mean_acc:.4f} ({sum(r.reward for r in results)}/{len(results)})")
        else:
            print("No results to compute accuracy.")

    if args.csv_file:
        import csv
        csv_dir = os.path.dirname(args.csv_file) or "."
        os.makedirs(csv_dir, exist_ok=True)
        with open(args.csv_file, 'w', newline='', encoding='utf-8') as cf:
            fieldnames = list(results[0].to_dict().keys()) if results else [
                'scene_index','question','answer_gt','model_output','action_letter','selected_images','verifier_question','verifier_answer','reward'
            ]
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        print(f"Also saved CSV to {args.csv_file}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Test action accuracy with verifier (expects JSONL with explicit image paths)")
    p.add_argument('--model-path', type=str, required=False, help='Required unless --use-gt-action is set')
    p.add_argument('--questions-file', type=str, required=True, help='JSONL file: each line must include an image path')
    p.add_argument('--output-file', type=str, required=True, help='JSONL output path')
    p.add_argument('--csv-file', type=str, default=None, help='Optional CSV output')
    p.add_argument('--verifier-model-path', type=str, default=os.getenv('VERIFIER_MODEL_PATH', 'qwen2.5vl:3b'))
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--limit', type=int, default=None, help='Process at most N questions (subset)')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--use-gt-action', action='store_true', help='Use ground-truth action from JSONL and skip action model inference')
    p.add_argument('--circular-eval', action='store_true', help='Evaluate each question with original and swapped A/B choices; count correct only if both are correct')
    return p

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    run(args)
