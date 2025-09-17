import argparse
import json
import os
from typing import List, Dict, Any
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from datetime import datetime

try:
    from tqdm import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except ImportError:  # fallback lightweight stub
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):  # noqa: D401
        return x


def extract_answer(text: str) -> str:
    import re
    if not isinstance(text, str):
        text = str(text)
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    ans = matches[-1] if matches else text
    ans = ans.strip().rstrip(".").lower()
    ans = " ".join(ans.split())
    return ans


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_messages(problem: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
    question = problem  # 문제 문장 그대로 사용 (이미 JSONL에 포함)
    # reasoning+answer 포맷 요구
    prompt = f"{question} First output the thinking process in <think> </think> and then the final answer in <answer> </answer>."
    content: List[Dict[str, Any]] = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def save_results_json(results_path: str, accuracy: float, results: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': accuracy,
            'total': len(results),
            'correct': sum(r['correct'] for r in results),
            'results': results
        }, f, ensure_ascii=False, indent=2)


def save_individual_samples(out_dir: str, results: List[Dict[str, Any]]):
    os.makedirs(out_dir, exist_ok=True)
    # 텍스트 요약 파일
    with open(Path(out_dir)/'samples.txt', 'w', encoding='utf-8') as ftxt:
        for i, r in enumerate(results, 1):
            ftxt.write(f"[{i}] image_paths={r['image_paths']}\n")
            ftxt.write(f"Q: {r['question']}\n")
            ftxt.write(f"GT: {r['gold_answer']}\n")
            ftxt.write(f"Pred: {r['pred_answer']}\n")
            ftxt.write(f"Correct: {r['correct']}\n")
            ftxt.write("---\n")
    # 이미지 복사(원본 경로 그대로 두고 symlink 가능)
    for i, r in enumerate(results, 1):
        for j, img_p in enumerate(r['image_paths']):
            if not img_p:
                continue
            try:
                img = Image.open(img_p).convert('RGB')
                img.save(Path(out_dir)/f"sample_{i:05d}_{j}.jpg")
            except Exception as e:
                print(f"[warn] fail save image {img_p}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Infer on saved validation JSONL and save detailed results.')
    parser.add_argument('--jsonl', required=True, help='Path to validation jsonl.')
    parser.add_argument('--model_path', required=True, help='Trained model checkpoint path.')
    parser.add_argument('--output_json', default='./logs/jsonl_eval_results.json', help='Aggregate JSON output path.')
    parser.add_argument('--output_samples_dir', default='./logs/jsonl_samples', help='Directory to save per-sample artifacts.')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--dtype', choices=['float32','float16','bfloat16'], default='bfloat16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of samples.')
    parser.add_argument('--save_every', type=int, default=0, help='If >0, save partial results every N processed samples.')
    parser.add_argument('--first100', action='store_true', help='If set, only evaluate the first 100 samples (applied after --limit if both given).')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    torch_dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except Exception:
        from transformers import AutoModelForCausalLM as VLModel  # fallback

    print(f"Loading model: {args.model_path}")
    model = VLModel.from_pretrained(args.model_path, torch_dtype=torch_dtype if device!='cpu' else torch.float32)
    if device!='cpu':
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    rows = load_jsonl(args.jsonl)
    if args.limit:
        rows = rows[:args.limit]
    if args.first100:
        rows = rows[:100]
    print(f"Loaded {len(rows)} rows")

    samples = []
    for r in rows:
        problem = r.get('problem') or r.get('question') or r.get('prompt') or ''
        solution = r.get('solution') or ''
        image_paths = r.get('image_path') or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        samples.append({'problem': problem, 'solution': solution, 'image_paths': image_paths})

    def batch(it, n):
        for i in range(0, len(it), n):
            yield it[i:i+n]

    all_results = []
    correct = 0
    total = 0

    def partial_save():
        """Save intermediate aggregate + sample text (images copied lazily)."""
        if not all_results:
            return
        # Recompute accuracy quickly
        _corr = sum(r['correct'] for r in all_results)
        _tot = len(all_results)
        _acc = (_corr / _tot * 100.0) if _tot else 0.0
        # Timestamped partial file names to avoid clobber if user wants history
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        interim_json = os.path.splitext(args.output_json)[0] + f"_partial_{_tot}_{ts}.json"
        save_results_json(interim_json, _acc, all_results)
        # For samples, just refresh main samples dir (idempotent). Avoid re-saving existing images.
        # We call save_individual_samples, which overwrites samples.txt but skip existing images by simple exists check.
        # Modify image saving logic inline (monkey patch) to skip existing.
        orig_open = Image.open
        def safe_save_samples(out_dir: str, results):
            os.makedirs(out_dir, exist_ok=True)
            # text summary
            with open(Path(out_dir)/'samples.txt', 'w', encoding='utf-8') as ftxt:
                for i, r in enumerate(results, 1):
                    ftxt.write(f"[{i}] image_paths={r['image_paths']}\n")
                    ftxt.write(f"Q: {r['question']}\n")
                    ftxt.write(f"GT: {r['gold_answer']}\n")
                    ftxt.write(f"Pred: {r['pred_answer']}\n")
                    ftxt.write(f"Correct: {r['correct']}\n")
                    ftxt.write("---\n")
            for i, r in enumerate(results, 1):
                for j, img_p in enumerate(r['image_paths']):
                    if not img_p:
                        continue
                    out_path = Path(out_dir)/f"sample_{i:05d}_{j}.jpg"
                    if out_path.exists():
                        continue
                    try:
                        img = orig_open(img_p).convert('RGB')
                        img.save(out_path)
                    except Exception:
                        pass
        safe_save_samples(args.output_samples_dir, all_results)
        print(f"[partial-save] {_tot} samples | acc={_acc:.2f}% | saved -> {interim_json}")

    iterator = batch(samples, args.batch_size)
    total_batches = (len(samples) + args.batch_size - 1)//args.batch_size
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=total_batches, desc='Infer')

    for chunk in iterator:
        images_group = []
        messages_group = []
        for s in chunk:
            imgs = []
            for p in s['image_paths']:
                if not p:
                    continue
                ap = p if os.path.isabs(p) else os.path.abspath(p)
                try:
                    imgs.append(Image.open(ap).convert('RGB'))
                except Exception as e:
                    print(f"[warn] cannot open image {ap}: {e}")
            messages = build_messages(s['problem'], imgs)
            messages_group.append(messages)
            images_group.append(imgs)

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_group]
        inputs = processor(text=texts, images=images_group, padding=True, return_tensors='pt')
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for s, out_text in zip(chunk, decoded):
            pred = extract_answer(out_text)
            gold = extract_answer(s['solution'])
            is_correct = 1 if pred == gold else 0
            correct += is_correct
            total += 1
            all_results.append({
                'image_paths': s['image_paths'],
                'question': s['problem'],
                'gold_answer': gold,
                'pred_answer': pred,
                'model_output': out_text,
                'correct': is_correct,
            })

            # Periodic intermediate save
            if args.save_every > 0 and (total % args.save_every == 0):
                partial_save()

        if not TQDM_AVAILABLE:
            if total % max(1, args.batch_size) == 0:  # fallback textual progress
                running_acc = (correct/total*100.0) if total else 0.0
                print(f"[progress] processed={total}/{len(samples)} acc={running_acc:.2f}%")

    accuracy = (correct / total * 100.0) if total else 0.0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    save_results_json(args.output_json, accuracy, all_results)
    save_individual_samples(args.output_samples_dir, all_results)
    print(f"Saved aggregate to {args.output_json}")
    print(f"Saved samples (images + summary) to {args.output_samples_dir}")


if __name__ == '__main__':
    main()
