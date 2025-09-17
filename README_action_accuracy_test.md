# Action Accuracy Test Script

This document describes how to run inference with action-based accuracy evaluation replicating the training reward logic.

## Script
`test_action_accuracy.py` (located alongside `grpo_jsonl.py`).

## Inputs
- **--model-path**: Fine-tuned model path or hub id (e.g. `./outputs/checkpoint-final` or `Qwen/Qwen2.5-VL-3B-Instruct`).
- **--questions-file**: JSON or JSON list file with fields: `question`, `answer`, `scene_index`.
- **--scenes-root**: Directory containing scene folders named `scene_000000`, `scene_000001`, ... Each folder must have at least `view_00.png` and optionally `view_01.png` ... `view_11.png`.
- **--output-file**: Path to write JSONL result lines.
- **--csv-file** (optional): CSV mirror of the JSONL.
- **--verifier-model-path** (optional): Model id/alias for verifier (defaults to env `VERIFIER_MODEL_PATH` or `qwen2.5vl:3b`).
- **--max-new-tokens**: Generation length for the tested model (default 128).
- **--verbose**: Print per-sample logs.

## Output JSONL Fields
Each line:
```json
{
  "scene_index": 0,
  "question": "What's the color of the small cube?",
  "answer_gt": "cyan",
  "model_output": "<think> ... </think><answer> A </answer>",
  "action_letter": "A",
  "selected_images": ["/path/scene_000000/view_00.png", "/path/scene_000000/view_01.png"],
  "verifier_answer": "cyan",
  "reward": 1.0
}
```
`action_letter` controls whether an additional view is appended for the verifier (A-K => view index 1..11; L => no extra view).

## Example
```bash
python -m open_r1.test_action_accuracy \
  --model-path ./finetuned-model \
  --questions-file /home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/test_question.json \
  --scenes-root /home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/test_scenes \
  --output-file ./action_accuracy_results.jsonl \
  --csv-file ./action_accuracy_results.csv \
  --verifier-model-path qwen2.5vl:3b \
  --verbose
```

## Notes
- Verifier failure -> reward 0.0 (conservative).
- Reward = exact match (case-insensitive, trailing period stripped) between verifier answer and ground truth.
- To speed up: set `CUDA_VISIBLE_DEVICES` appropriately; ensure bf16 supported GPU for memory savings.
- If your fine-tuned model already emits `<answer>` blocks the extraction of action letter will prefer them.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| `Primary view not found` | Missing `view_00.png` | Ensure dataset export produced the base view. |
| Verifier OOM | Model too large for GPU | Use `qwen2.5vl:3b` alias or smaller verifier. |
| All rewards 0 | Action letter wrong or verifier mismatch | Inspect `model_output` and `verifier_answer` fields. |

---
Generated automatically to accompany the evaluation script.
