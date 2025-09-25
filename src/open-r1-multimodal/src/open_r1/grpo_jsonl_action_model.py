# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration, AutoModelForVision2Seq
import torch

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json

from open_r1.vlm_modules import *

from typing import Tuple
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer

from openai import OpenAI
from open_r1.grpo_jsonl import accuracy_reward, format_reward, cosine_rewards, \
        repetition_rewards, get_vlm_module, GRPOScriptArguments, GRPOModelConfig, initialize_tokenizer
logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, monkey_patch_torch_load
monkey_patch_qwen2_5vl_flash_attn()    
monkey_patch_torch_load()

tokenizer = None

# ---------------- Action-based reward using gt_action ---------------- #
# Action prediction model reward using gt_action comparison
ACTION_LETTERS = {
    'A': 1,  # view_01.png
    'B': 2,  # view_02.png
}

def _extract_final_answer_letter(text: str):
    """학생 모델 출력에서 최종 액션(A 또는 B) 추출.

    규칙:
      - 마지막 <answer>...</answer> 블록을 찾는다. 없으면 None 반환
      - 내용이 단일 문자 'A' 또는 'B' (대소문자 허용, 끝의 마침표 1개 허용) 이면 해당 대문자 반환
      - 그 외는 None
    """
    answer_blocks = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if not answer_blocks:
        return None
    candidate_raw = answer_blocks[-1].strip()
    if candidate_raw.endswith('.') and candidate_raw.count('.') == 1:
        candidate_core = candidate_raw[:-1].strip()
    else:
        candidate_core = candidate_raw
    candidate_up = candidate_core.upper()
    return candidate_up if candidate_up in ACTION_LETTERS else None

def action_accuracy_reward(completions, solution, **kwargs):
    """Action prediction reward using gt_action comparison.
    
    This reward function compares the predicted action (A or B) with the ground truth action
    from the dataset's 'gt_action' field. No verifier model is used - direct comparison only.
    
    Reward = 1.0 if predicted action matches gt_action, 0.0 otherwise
    """
    rewards = []
    contents = [c[0]["content"] for c in completions]
    
    for i, (content, sol) in enumerate(zip(contents, solution)):
        # Extract predicted action from model output
        letter = _extract_final_answer_letter(content)
        
        # Get ground truth action from kwargs
        gt_actions_arg = kwargs.get('gt_action')
        gt_action_item = None
        if isinstance(gt_actions_arg, list) and len(gt_actions_arg) == len(contents):
            gt_action_item = gt_actions_arg[i]
        elif isinstance(gt_actions_arg, list) and len(gt_actions_arg) > 0:
            gt_action_item = gt_actions_arg[0]
        
        # Calculate reward: 1.0 if actions match, 0.0 otherwise
        if letter and gt_action_item:
            reward = 1.0 if letter.upper() == gt_action_item.upper() else 0.0
        else:
            reward = 0.0
        
        rewards.append(reward)
        
        # Debug logging
        try:
            print(f"[action_accuracy][debug] gt_action={gt_action_item} predicted_action={letter} reward={reward:.3f}")
        except Exception:
            pass
            
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "debug_action_accuracy.txt")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace('.txt','_action_accuracy.txt'), 'a', encoding='utf-8') as f:
                f.write(f"----- {current_time} action_accuracy reward: {reward} -----\n")
                f.write(f"predicted_action: {letter}, gt_action: {gt_action_item}\n")
                f.write(f"student_raw: {content}\n")
                
    return rewards


def action_match_reward(completions, solution, **kwargs):
    """Action prediction reward using gt_action comparison.
    
    This is an alias for action_accuracy_reward - both functions do the same thing:
    compare predicted action with ground truth action.
    """
    return action_accuracy_reward(completions, solution, **kwargs)


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": cosine_rewards,
    "repetition": repetition_rewards,
    "action_accuracy": action_accuracy_reward,
    "action_match": action_match_reward,
    "multi_choice_reward": action_accuracy_reward  # Alias for action prediction reward
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.task_type)

    # Get reward functions 
    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in script_args.reward_funcs]
    else:
        reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if script_args.action_mapping == 'ba':
        ACTION_LETTERS.update({'A': 2, 'B': 1})  # A=right, B=right
    elif script_args.action_mapping == 'ab':
        ACTION_LETTERS.update({'A': 1, 'B': 2})  # A=left, B=right

    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                print
                if 'image' in item:
                    if isinstance(item['image'], str):
                        # If path is absolute, keep as-is; otherwise join with folder
                        img_path = item['image'] if os.path.isabs(item['image']) else os.path.join(image_folder, item['image'])
                        item['image_path'] = [img_path]
                        del item['image'] # remove the image column so that it can be loaded later
                    elif isinstance(item['image'], list):
                        # if the image is a list, then it is a list of images (for multi-image input)
                        paths = []
                        for image in item['image']:
                            paths.append(image if os.path.isabs(image) else os.path.join(image_folder, image))
                        item['image_path'] = paths
                        del item['image'] # remove the image column so that it can be loaded later
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                # Remove immediate image loading
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                # Pass through optional gt_action for verifier mapping only
                if 'gt_action' in item and isinstance(item['gt_action'], str):
                    item['gt_action'] = item['gt_action'].strip().upper()
                
                # Handle solution that could be a float or string
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    # If it's a float or other non-string type, keep it as is
                    item['solution'] = str(solution_value)
                
                del item['conversations']
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            assert all(os.path.exists(p) for p in example['image_path']), f"Image paths do not exist: {example['image_path']}"
            # Don't load image here, just store the path
            return {
                'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
                'problem': example['problem'],
                'gt_action': example.get('gt_action'),
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'gt_action': example.get('gt_action'),
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio,
            seed=script_args.val_split_seed
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

        # Optionally persist validation split
        if script_args.save_validation_path:
            val_ds = splits.get('validation')
            save_path = script_args.save_validation_path
            try:
                if save_path.lower().endswith(".jsonl") or save_path.lower().endswith(".json"):
                    # Prefer Dataset.to_json; fallback to manual JSONL
                    try:
                        # lines=True ensures JSON Lines when supported
                        val_ds.to_json(save_path, lines=True, force_ascii=False)
                    except Exception:
                        import json as _json
                        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                        with open(save_path, 'w', encoding='utf-8') as _f:
                            for row in val_ds:
                                _f.write(_json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"Saved validation split to JSON at: {save_path}")
                else:
                    # Treat as Arrow dataset directory
                    os.makedirs(save_path, exist_ok=True)
                    val_ds.save_to_disk(save_path)
                    print(f"Saved validation split (Arrow) to: {save_path}")
            except Exception as e:
                print(f"[warn] Failed to save validation split to {save_path}: {e}")
    elif script_args.save_validation_path:
        print("[warn] save_validation_path is set but val_split_ratio == 0; no validation split to save.")

    # Select trainer class based on vlm_trainer argument
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    
    # Print reward functions being used
    print("Reward functions:", [func.__name__ for func in reward_funcs])
    
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
