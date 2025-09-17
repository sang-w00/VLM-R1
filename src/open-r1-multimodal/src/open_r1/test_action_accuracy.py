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

ACTION_LETTERS = {chr(ord('A') + i): i + 1 for i in range(11)}  # A-K -> 1..11
ACTION_LETTERS['L'] = 0  # L means no rotation / no extra view

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

def extract_final_answer_letter(text: str) -> str:
    """Replicates training logic: only trust the last <answer> block and return a single A-L letter.

    Rules:
      - If no <answer> tag: return 'L'
      - Take last <answer> content, strip; allow one trailing '.' (e.g., 'A.')
      - If (uppercased, after stripping trailing '.') is exactly one char in A-L -> return that uppercase letter
      - Otherwise return 'L'
    """
    blocks = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return 'L'
    candidate_raw = blocks[-1].strip()
    if candidate_raw.endswith('.') and candidate_raw.count('.') == 1:
        candidate_core = candidate_raw[:-1].strip()
    else:
        candidate_core = candidate_raw
    candidate_up = candidate_core.upper()
    if re.fullmatch(r'[A-L]', candidate_up):
        return candidate_up
    return 'L'

def build_additional_view(primary_path: str, letter: str) -> Optional[str]:
    if letter not in ACTION_LETTERS or letter == 'L':
        return None
    idx = ACTION_LETTERS[letter]
    base_dir = os.path.dirname(primary_path)
    candidate = os.path.join(base_dir, f"view_{idx:02d}.png")
    return candidate if os.path.exists(candidate) else None

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
    pil_images = [Image.open(p).convert('RGB') for p in image_paths]
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in pil_images],
            {"type": "text", "text": f"{question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. The text between <answer> and </answer> must be exactly one uppercase letter from A to L (A|B|C|D|E|F|G|H|I|J|K|L). No spaces, words, punctuation, or additional characters are allowed. If uncertain, choose the best single letter. Do not output anything after </answer>."}
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
                entries.append({
                    'scene_index': scene_index if isinstance(scene_index, int) else int(scene_index) if str(scene_index).isdigit() else len(entries),
                    'question': question_text,
                    'answer': answer_text,
                    'image': image_path,
                    'raw_prompt': raw_prompt,
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
        }

def run(args):
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

    for q in questions:
        # Normalized entries from JSONL loader expected (must contain image path)
        scene_index = q.get('scene_index') if isinstance(q, dict) else q['scene_index']
        question_text = q.get('question') if isinstance(q, dict) else q['question']  # extracted concise question for verifier
        gt_answer = q.get('answer') or q.get('answer_gt') or q.get('gt')
        primary_img = q.get('image') if isinstance(q, dict) else q['image']
        raw_prompt = q.get('raw_prompt', question_text)
        if not primary_img:
            print(f"[warn] skip scene {scene_index}: missing image path (provide in JSONL)")
            continue
        if not os.path.exists(primary_img):
            print(f"[warn] image not found for scene {scene_index}: {primary_img}")
            continue

        # 1. Generate model output on primary view only using full raw prompt (includes instruction/options)
        model_output = generate_model_answer(model, processor, [primary_img], raw_prompt, max_new_tokens=args.max_new_tokens)

        # 2. Extract action and build additional view (not fed back to model, only for verifier)
        action_letter = extract_final_answer_letter(model_output)
        add_view = build_additional_view(primary_img, action_letter)
        selected_for_verifier = [primary_img] + ([add_view] if add_view else [])

        # 3. Verifier inference uses only the concise question
        verifier_question = question_text
        verifier_ans, _verifier_raw = verifier_answer(selected_for_verifier, verifier_question, args.verifier_model_path)

        # 4. Reward using verifier answer vs ground truth (mirrors training logic)
        if verifier_ans is None:
            reward = 0.0
        else:
            reward = default_accuracy_reward(f"<answer>{verifier_ans}</answer>", f"<answer>{gt_answer}</answer>")

        entry = ResultEntry(
            scene_index=int(scene_index),
            question=question_text,
            answer_gt=gt_answer,
            model_output=model_output,
            action_letter=action_letter,
            selected_images=selected_for_verifier,
            verifier_answer=verifier_ans,
            verifier_question=verifier_question,
            reward=reward,
        )
        results.append(entry)
        if args.verbose:
            print(f"[test] scene={scene_index} action={action_letter} reward={reward:.3f} verifier_q='{verifier_question}' verifier_ans='{verifier_ans}' gt='{gt_answer}'")

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(results)} results to {args.output_file}")

    # Final accuracy
    if results:
        mean_acc = sum(r.reward for r in results) / len(results)
        print(f"Final accuracy: {mean_acc:.4f} ({sum(r.reward for r in results)}/{len(results)})")
    else:
        print("No results to compute accuracy.")

    if args.csv_file:
        import csv
        os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
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
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--questions-file', type=str, required=True, help='JSONL file: each line must include an image path')
    p.add_argument('--output-file', type=str, required=True, help='JSONL output path')
    p.add_argument('--csv-file', type=str, default=None, help='Optional CSV output')
    p.add_argument('--verifier-model-path', type=str, default=os.getenv('VERIFIER_MODEL_PATH', 'qwen2.5vl:3b'))
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--limit', type=int, default=None, help='Process at most N questions (subset)')
    p.add_argument('--verbose', action='store_true')
    return p

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    run(args)
