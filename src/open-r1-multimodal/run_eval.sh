# evaluate grpo fine-tuned model
CUDA_VISIBLE_DEVICES=0 python src/open_r1/test_action_accuracy.py \
--model-path /home/daehyeonchoi/embodied/VLM-R1/output/checkpoint-892 \
--questions-file /home/daehyeonchoi/embodied/procthor-10k/outputs/procthor_vqa_data_val/data.jsonl \
--output-file /home/daehyeonchoi/embodied/VLM-R1/runs/infer/procthor_val_grpo/qwen_base_model/output.jsonl \
--verifier-model-path qwen2.5vl:3b \
--circular-eval \
--verbose

# evaluate zeroshot model (3b)
CUDA_VISIBLE_DEVICES=0 python src/open_r1/test_action_accuracy_zeroshot.py \
--model-path qwen2.5vl:3b \
--questions-file /home/daehyeonchoi/embodied/procthor-10k/outputs/procthor_vqa_data_val/data.jsonl \
--output-file /home/daehyeonchoi/embodied/VLM-R1/runs/infer/procthor_val_grpo/qwen_base_3b/output.jsonl \
--verifier-model-path qwen2.5vl:3b \
--verbose


# evaluate zeroshot model (7b)
CUDA_VISIBLE_DEVICES=1 python src/open_r1/test_action_accuracy_zeroshot.py \
--model-path qwen2.5vl:7b \
--questions-file /home/daehyeonchoi/embodied/procthor-10k/outputs/procthor_vqa_data_val/data.jsonl \
--output-file /home/daehyeonchoi/embodied/VLM-R1/runs/infer/procthor_val_grpo/qwen_base_7b/output.jsonl \
--verifier-model-path qwen2.5vl:3b \
--verbose


