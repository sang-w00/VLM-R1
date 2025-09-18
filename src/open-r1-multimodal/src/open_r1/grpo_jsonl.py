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

logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, monkey_patch_torch_load
monkey_patch_qwen2_5vl_flash_attn()    
monkey_patch_torch_load()

tokenizer = None
verifier_model = None
verifier_processor = None

VERIFIER_MODEL_PATH = os.getenv("VERIFIER_MODEL_PATH", "qwen2.5vl:3b")  # alias allowed (e.g., qwen2.5vl:3b, qwen2.5vl:7b)

ALIAS_MAP = {
    "qwen2.5vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5_vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5_vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}

def resolve_model_path(raw_path: str) -> str:
    key = raw_path.strip().lower()
    return ALIAS_MAP.get(key, raw_path)

def initialize_verifier():
    """Lazy load frozen verifier model (Qwen2.5-VL) for action-based accuracy reward.

    환경변수 VERIFIER_MODEL_PATH 로 모델 경로 재정의 가능. 지원 alias: qwen2.5vl:3b, qwen2.5vl:7b
    """
    global verifier_model, verifier_processor
    if verifier_model is None:
        target_path = resolve_model_path(VERIFIER_MODEL_PATH)
        tried = []
        candidates = [target_path]
        if 'Instruct' not in target_path and '-Instruct' not in target_path:
            if target_path.endswith('-7B') or target_path.endswith('-3B'):
                candidates.append(target_path + '-Instruct')
        for cand in candidates:
            try:
                from transformers import AutoProcessor
                verifier_processor = AutoProcessor.from_pretrained(cand, trust_remote_code=True)
                # Decide which loader
                if any(tag in cand.lower() for tag in ['2.5-vl', '2_5-vl', '2.5_vl', '2_5_vl']):
                    loader_cls = AutoModelForVision2Seq
                else:
                    loader_cls = Qwen2VLForConditionalGeneration
                verifier_model = loader_cls.from_pretrained(
                    cand,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
                verifier_model.eval()
                for p in verifier_model.parameters():
                    p.requires_grad_(False)
                print(f"[action_accuracy] Loaded verifier model: {cand} via {loader_cls.__name__}")
                break
            except Exception as e:
                print(f"[action_accuracy] Failed to load {cand}: {e}")
                tried.append(cand)
                verifier_model = None
                verifier_processor = None
        if verifier_model is None:
            print(f"[action_accuracy] All load attempts failed: {tried}")
    return verifier_model, verifier_processor

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    val_split_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed used for train/validation split (datasets.train_test_split)."},
    )
    save_validation_path: Optional[str] = field(
        default=None,
        metadata={"help": "If set, save validation split to this path. Use a directory for Arrow (save_to_disk) or a file ending with .jsonl/.json for JSON."},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )

def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]

def evaluate_answer_similarity(student_answer, ground_truth):
    """Use llm to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": "You are a evaluation expert. First, analyze the student's response to identify and extract their final answer. Then, compare the extracted answer with the correct solution. Output ONLY '1.0' if the extracted answer matches the correct solution in meaning, or '0.0' if the student's response does not contain a clear or correct answer. No other output is allowed."
                },
                {
                    "role": "user",
                    "content": f"Student's response: {student_answer}\nCorrect solution: {ground_truth}\nOutput only 1.0 or 0.0:"
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return float(result)
    
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if student_answer ==ground_truth else 0.0

def llm_reward(content, sol, **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return evaluate_answer_similarity(student_answer, ground_truth)

def mcq_reward(content, sol, **kwargs):
    # For multiple choice, extract and compare choices
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    has_choices = extract_choice(ground_truth)
    correct_choice = has_choices.upper() if has_choices else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_choice = extract_choice(student_answer)
    if student_choice:
        reward = 1.0 if student_choice == correct_choice else 0.0
    else:
        reward = 0.0

    return reward


def yes_no_reward(content, sol, **kwargs):
    content = content.lower()
    sol = sol.lower()

    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()

    ground_yes_no = re.search(r'(yes|no)', ground_truth)
    ground_yes_no = ground_yes_no.group(1) if ground_yes_no else ''
    student_yes_no = re.search(r'(yes|no)', student_answer)
    student_yes_no = student_yes_no.group(1) if student_yes_no else ''

    reward = 1.0 if ground_yes_no == student_yes_no else 0.0

    return reward

# score_type: 0 for mAP, 1 for mAP 50
def calculate_map(pred_bbox_list, gt_bbox_list, score_type=0):
    # Calculate mAP

    # Initialize COCO object for ground truth
    gt_json = {"annotations": [], "images": [], "categories": []}
    gt_json["images"] = [{
        "id": 0,
        "width": 2048,
        "height": 2048,
        "file_name": "image_0.jpg"
    }]

    gt_json["categories"] = []

    cats2id = {}
    cat_count = 0
    for idx, gt_bbox in enumerate(gt_bbox_list):
        if gt_bbox["label"] not in cats2id:
            cats2id[gt_bbox["label"]] = cat_count
            gt_json["categories"].append({
                "id": cat_count,
                "name": gt_bbox["label"]
            })
            cat_count += 1
        
        gt_json["annotations"].append({
            "id": idx+1,
            "image_id": 0,
            "category_id": cats2id[gt_bbox["label"]],
            "bbox": [gt_bbox["bbox_2d"][0], gt_bbox["bbox_2d"][1], gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0], gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]],
            "area": (gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0]) * (gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]),
            "iscrowd": 0
        })
    coco_gt = COCO(gt_json)

    dt_json = []
    for idx, pred_bbox in enumerate(pred_bbox_list):
        try:
            dt_json.append({
                "image_id": 0,
                "category_id": cats2id[pred_bbox["label"]],
                "bbox": [pred_bbox["bbox_2d"][0], pred_bbox["bbox_2d"][1], pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0], pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1]],
                "score": 1.0,
                "area": (pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0]) * (pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1])
            })
        except:
            pass
    
    if len(dt_json) == 0:
        return 0.0
    
    coco_dt = coco_gt.loadRes(dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[score_type]

def map_reward(content, sol, length_reward=False, score_type=0, **kwargs):
    """
    Calculate mean average precision (mAP) reward between predicted and ground truth bounding boxes.
    
    Args:
        content (str): String containing predicted bounding boxes in JSON format
        sol (str): String containing ground truth bounding boxes in JSON format
        length_reward (bool, optional): Whether to include length penalty in reward calculation. Defaults to False.
        score_type (int, optional): Type of COCO evaluation metric to use. Defaults to 0 (mAP).
        **kwargs: Additional keyword arguments
        
    Returns:
        float: mAP reward score between 0 and 1. If length_reward is True, the score is multiplied by a length penalty factor.
    """
    # Extract JSON content between ```json tags
    pattern = r'```json(.*?)```'
    json_match = re.findall(pattern, sol, re.DOTALL)
    bbox_json = json_match[-1].strip() if json_match else None

    # Parse ground truth JSON to get bbox list
    gt_bbox_list = []
    if bbox_json:
        bbox_data = json.loads(bbox_json)
        gt_bbox_list = [item for item in bbox_data]
    
    # Parse predicted JSON to get bbox list
    pred_bbox_list = []
    json_match = re.findall(pattern, content, re.DOTALL)
    if json_match:
        try:
            bbox_data = json.loads(json_match[-1].strip())
            pred_bbox_list = [item for item in bbox_data]
        except:
            # Return empty list if JSON parsing fails
            pred_bbox_list = []

    # Calculate mAP if both prediction and ground truth exist
    if len(pred_bbox_list) > 0 and len(gt_bbox_list) > 0:
        bbox_reward = calculate_map(pred_bbox_list, gt_bbox_list, score_type=score_type)
    elif len(pred_bbox_list) == 0 and len(gt_bbox_list) == 0:
        bbox_reward = 1.0
    else:
        bbox_reward = 0.0
    
    if length_reward:
        # Calculate length penalty based on ratio of ground truth to predicted bounding boxes
        gt_length = len(gt_bbox_list)
        pred_length = len(pred_bbox_list)
        # Full score if prediction has fewer boxes than ground truth, otherwise penalize proportionally
        length_score = 1.0 if gt_length >= pred_length else gt_length/pred_length
        return bbox_reward * length_score
    else:
        return bbox_reward

def od_reward(content, sol, score_type=0, **kwargs):
    """
    Calculate reward for object detection task by comparing predicted and ground truth answers.
    
    Args:
        content (str): Model's predicted answer containing bounding box annotations
        sol (str): Ground truth answer containing bounding box annotations 
        score_type (int): Type of COCO evaluation metric to use (default: 0 for mAP)
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Reward score between 0 and 1 based on mAP between predicted and ground truth boxes
    """
    # Pattern to extract content between <answer> tags
    match_pattern = r'<answer>(.*?)</answer>'

    # Extract ground truth answer
    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None

    # Extract predicted answer (using last match if multiple)
    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    # Return 0 if no prediction
    if student_answer is None:
        return 0.0
    # Return 1 if both prediction and ground truth are None
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    # Otherwise calculate mAP between prediction and ground truth
    else:
        return map_reward(student_answer, ground_truth, score_type=score_type)

def odLength_reward(content, sol, **kwargs):
    """
    Calculate reward for object detection task with length penalty.
    
    Args:
        content (str): Model's predicted answer containing bounding box annotations
        sol (str): Ground truth answer containing bounding box annotations
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Reward score between 0 and 1 based on mAP and length penalty
    """
    # Pattern to extract content between <answer> tags
    match_pattern = r'<answer>(.*?)</answer>'

    # Extract ground truth answer
    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None
    # Extract predicted answer (using last match if multiple)
    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    # Return 0 if no prediction
    if student_answer is None:
        return 0.0
    # Return 1 if both prediction and ground truth are None
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    # Calculate mAP with length penalty
    else:
        bbox_reward = map_reward(student_answer, ground_truth, length_reward=True, score_type=0)
        return bbox_reward

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union


def detection_score(content, sol, iou_threshold=0.5, alpha=0.7, beta=0.0, gamma=0.3):
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, clean_text(content), re.DOTALL)
    content_bbox_json = json_match.group(1).strip() if json_match else None
    if content_bbox_json:
        try:
            bbox_data = json.loads(content_bbox_json)
            pred_boxes = [item for item in bbox_data]
        except:
            pred_boxes = []

    else:
        pred_boxes = []

    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, clean_text(sol), re.DOTALL)
    sol_bbox_json = json_match.group(1).strip() if json_match else None
    if sol_bbox_json:
        bbox_data = json.loads(sol_bbox_json)
        gt_boxes = [item for item in bbox_data]
    else:
        gt_boxes = []

    """
    Calculate the comprehensive score for object detection
    
    Parameters:
        pred_boxes: List of predicted boxes, each element is in the format {"bbox_2d": [x1, y1, x2, y2], "label": "category name"}
        gt_boxes: List of ground truth boxes, each element is in the format {"bbox_2d": [x1, y1, x2, y2], "label": "category name"}
        iou_threshold: IoU threshold, default is 0.5
        alpha: Position accuracy weight, default is 0.7
        beta: Label accuracy weight, default is 0.0
        gamma: Completeness weight (penalty for missed/false detections), default is 0.3
        
    Returns:
        Comprehensive score, ranging from [0.0, 1.0]
    """
    # Handle edge cases
    if len(gt_boxes) == 0:
        return 1.0 if not pred_boxes else 0.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    # Initialize matching results
    matches = []  # Store matched pairs of predicted and ground truth boxes
    unmatched_preds = list(range(len(pred_boxes)))  # Indices of unmatched predicted boxes
    unmatched_gts = list(range(len(gt_boxes)))  # Indices of unmatched ground truth boxes
    
    # Calculate IoU matrix between all predicted and ground truth boxes
    iou_matrix = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        iou_row = []
        for gt_idx, gt_box in enumerate(gt_boxes):
            try:
                curr_iou = iou(pred_box["bbox_2d"], gt_box["bbox_2d"])
            except:
                curr_iou = 0.0
            iou_row.append(curr_iou)
        iou_matrix.append(iou_row)
    
    # Greedy matching: find the best match for each predicted box
    while unmatched_preds and unmatched_gts:
        # Find the maximum IoU
        max_iou = -1
        max_pred_idx = -1
        max_gt_idx = -1
        
        for pred_idx in unmatched_preds:
            for gt_idx in unmatched_gts:
                curr_iou = iou_matrix[pred_idx][gt_idx]
                if curr_iou > max_iou:
                    max_iou = curr_iou
                    max_pred_idx = pred_idx
                    max_gt_idx = gt_idx
        
        # Stop matching if the maximum IoU is below the threshold
        if max_iou < iou_threshold:
            break
        
        # Record matching results
        try:
            pred_label = pred_boxes[max_pred_idx]["label"].lower()
        except:
            pred_box = ""
        try:
            gt_label = gt_boxes[max_gt_idx]["label"].lower()
        except:
            gt_label = ""
        label_correct = (pred_label == gt_label)
        
        if label_correct:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": max_iou,
                "label_correct": label_correct
            })
        else:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": 0,
                "label_correct": label_correct
            })
        
        # Remove matched boxes from the unmatched list
        unmatched_preds.remove(max_pred_idx)
        unmatched_gts.remove(max_gt_idx)
    
    # Calculate position accuracy score (average IoU)
    position_score = sum(m["iou"] for m in matches) / len(gt_boxes) if matches else 0.0
    
    # Calculate label accuracy score
    label_score = sum(1.0 for m in matches if m["label_correct"]) / len(gt_boxes) if matches else 0.0
    
    # Calculate completeness score (considering missed and false detections)
    # Miss rate = number of unmatched ground truth boxes / total number of ground truth boxes
    # False alarm rate = number of unmatched predicted boxes / total number of predicted boxes
    miss_rate = len(unmatched_gts) / len(gt_boxes)
    false_alarm_rate = len(unmatched_preds) / len(pred_boxes) if pred_boxes else 0.0
    
    # Completeness score = 1 - (miss rate + false alarm rate) / 2
    completeness_score = 1.0 - (miss_rate + false_alarm_rate) / 2.0
    
    # Calculate the final comprehensive score
    final_score = (
        alpha * position_score + 
        beta * label_score + 
        gamma * completeness_score
    ) / (alpha + beta + gamma)

    return final_score

def cosine_reward(content, tokenizer, acc_reward, **kwargs):
    #https://arxiv.org/abs/2502.03373
    min_len_value_wrong = 0.0
    max_len_value_wrong = -0.5
    min_len_value_correct = 1.0
    max_len_value_correct = 0.5
    cosine_max_len = 1024

    # processing_class = AutoProcessor.from_pretrained(model_path)
    # tokenizer = processing_class.tokenizer
    
    gen_len = len(tokenizer.encode(content))
    acc_reward = 1.0
    is_correct = acc_reward >= 0.7
    
    if is_correct:
        # Swap min/max for correct answers
        min_value = max_len_value_correct
        max_value = min_len_value_correct
    else:
        min_value = min_len_value_wrong
        max_value = max_len_value_wrong

    reward = max_value - (max_value - min_value) * (1 - math.cos(gen_len * math.pi / cosine_max_len)) / 2

    return reward

def repetition_reward(content, **kwargs):
    max_penalty = -1.0

    if content == '':
        return 0.0

    # First, try to extract explicitly marked JSON sections
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, content, re.DOTALL)
    
    if json_match:
        bbox_json = json_match.group(1).strip()
    else:
        # If no explicitly marked JSON is found, try to find any possible JSON sections
        pattern = r'```(.*?)```'
        json_match = re.search(pattern, content, re.DOTALL)
        bbox_json = json_match.group(1).strip() if json_match else None
        
        # If still not found, try to find possible JSON array sections
        if not bbox_json:
            pattern = r'\[\s*{.*?"bbox_2d".*?"label".*?}\s*\]'
            json_match = re.search(pattern, content, re.DOTALL)
            bbox_json = json_match.group(0) if json_match else None
    
    # Try to parse JSON data
    if bbox_json:
        try:
            # Try direct parsing
            data = json.loads(bbox_json)
        except json.JSONDecodeError:
            try:
                # If direct parsing fails, try using json_repair to repair
                repaired_json = repair_json(bbox_json)
                data = json.loads(repaired_json)
            except:
                # If repair also fails, switch to plain text processing
                data = None
        if data and isinstance(data, list):
            # Ensure data is in list format
            try:
                # For JSON data, set ngram_size to 1
                ngram_size = 1
                # Combine 'bbox_2d' and 'label' of each object into a string
                items = []
                for item in data:
                    if 'bbox_2d' in item and 'label' in item:
                        items.append(f"{item['bbox_2d']}_{item['label']}")
                
                @staticmethod
                def zipngram(text: list, ngram_size: int):
                    return zip(*[text[i:] for i in range(ngram_size)])
                
                ngrams = set()
                total = 0

                for ng in zipngram(items, ngram_size):
                    ngrams.add(ng)
                    total += 1

                if total == 0:
                    return 0.0

                scaling = 1 - len(ngrams) / total
                reward = scaling * max_penalty

                return reward
            except KeyError:
                # If necessary keys are missing, switch to plain text processing
                pass
    
    # If no JSON section is found or JSON processing fails, treat as plain text
    ngram_size = 6
    
    if len(content.split()) < ngram_size:
        return 0.0
    
    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    
    ngrams = set()
    total = 0

    for ng in zipngram(content, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    return reward


def repetition_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        reward = repetition_reward(content)
        rewards.append(reward)


        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 0.0:  # this condition can be changed for debug
                with open(log_path+"_repetition.txt", "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")     



    return rewards


def cosine_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        clean_content = clean_text(content)
        sol = clean_text(sol)
        if sol == "none":
            if clean_content == "none":
                acc_reward = 1.0
            else:
                acc_reward = 0.0
        else:
            acc_reward = detection_score(clean_content, sol)
        reward = cosine_reward(content, tokenizer, acc_reward)
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <=1.0:  # this condition can be changed for debug
                with open(log_path+"_cosine.txt", "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")   

    return rewards

def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None
def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)
def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def all_match_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return 1.0 if content == sol else 0.0

def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
        # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    
    # Try symbolic verification first for numeric answers
    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try: 
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            has_choices = extract_choice(ground_truth)
            
            if has_numbers:
                # For numeric answers, use exact matching
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                # For text answers, use fuzzy matching
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == "mcq":
            reward = mcq_reward(content, sol)
        elif accu_reward_method == 'yes_no':
            reward = yes_no_reward(content, sol)
        elif accu_reward_method == 'llm':
            reward = llm_reward(content, sol)
        elif accu_reward_method == 'map':
            reward = map_reward(content, sol)
        elif accu_reward_method == 'math':
            reward = math_reward(content, sol)
        elif accu_reward_method == 'weighted_sum':
            clean_content = clean_text(content)
            sol = clean_text(sol)
            if sol == "none":
                if clean_content == "none":
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = detection_score(clean_content, sol)
        elif accu_reward_method == 'od_ap':
            reward = od_reward(content, sol)
        elif accu_reward_method == 'od_ap50':
            reward = od_reward(content, sol, score_type=1)
        elif accu_reward_method == 'odLength':
            reward = odLength_reward(content, sol)
        elif accu_reward_method == 'all_match':
            reward = all_match_reward(content, sol)
        else:
            reward = default_accuracy_reward(content, sol)  
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")     

        
    return rewards


def format_reward(completions, **kwargs):
    """Format reward with three levels:
    - 1.0: Both <think>...</think><answer>...</answer> 태그 구조가 전체 문자열을 구성하고, <answer> 내용이 정확히 단일 문자 A 또는 B (대소문자 허용)
    - 0.5: 태그 구조(<think> + <answer>)는 올바르게 닫혀 있고 전체 문자열이지만, <answer> 내용이 단일 A-L 문자가 아님
    - 0.0: 위 조건을 만족하지 못함 (태그 누락, 순서 오류, 바깥 여분 텍스트 등)

    허용 전체 패턴 (공백은 유연):
        <think> ... </think><answer> X </answer>
    여기서 X 는 단일 A 또는 B 일 때 1.0, 그 외면 0.5.
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    # 1단계: 태그 구조 매칭 (answer 내용 캡쳐)
    structure_pattern = r"<think>.*?</think>\s*<answer>\s*(.*?)\s*</answer>\s*$"
    rewards = []
    debug_infos = []
    for content in completion_contents:
        stripped = content.strip()
        m = re.fullmatch(structure_pattern, stripped, re.DOTALL | re.IGNORECASE)
        if not m:
            rewards.append(0.0)
            debug_infos.append((content, None, 0.0, 'no-structure'))
            continue
        answer_raw = m.group(1).strip()
        # 단일 A 또는 B 여부 (대소문자 허용)
        if re.fullmatch(r'[ABab]', answer_raw):
            rewards.append(1.0)
            debug_infos.append((content, answer_raw, 1.0, 'single-letter'))
        else:
            rewards.append(0.5)
            debug_infos.append((content, answer_raw, 0.5, 'structure-only'))

    if os.getenv("DEBUG_MODE") == "true":
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        log_path = os.getenv("LOG_PATH", "debug.txt")
        try:
            with open(log_path.replace('.txt', '_format.txt'), 'a', encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for original, ans, rew, reason in debug_infos:
                    f.write(f"reward={rew} reason={reason} answer_raw={ans}\n")
                    f.write(f"Content: {original}\n")
        except Exception:
            pass

    return rewards


# ---------------- New Reward: action-based verification accuracy ---------------- #
# (UPDATED) 선택지는 이제 A 또는 B 두 개만 사용.
# 기존 A-L 매핑에서 축소: A -> view_01, B -> view_02 로 해석.
# 추가 뷰가 필요 없는(이전의 'L') 개념은 더 이상 사용하지 않고, 잘못된/미존재 답은 추가 뷰 없이 진행.
ACTION_LETTERS = {
    'A': 1,  # view_01.png
    'B': 2,  # view_02.png
}

def _view_index_from_path(path: str):
    m = re.search(r'view_(\d{2})', path)
    return int(m.group(1)) if m else None

def _extract_final_answer_letter(text: str):
    """학생 모델 출력에서 최종 액션(A 또는 B) 추출.

    규칙:
      - 마지막 <answer>...</answer> 블록을 찾는다. 없으면 None 반환 (추가 뷰 없음)
      - 내용이 단일 문자 'A' 또는 'B' (대소문자 허용, 끝의 마침표 1개 허용) 이면 해당 대문자 반환
      - 그 외는 None (추가 뷰 없음)
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

def _scene_index_from_path(path: str):
    m = re.search(r'scene_(\d{6})', path)
    return int(m.group(1)) if m else None

def _build_additional_view(original_path: str, letter: str | None, mapping: dict[str, int] | None = None):
    mapping = mapping or ACTION_LETTERS
    if letter is None or letter not in mapping:
        return None
    idx = mapping[letter]
    base_dir = os.path.dirname(original_path)
    candidate = os.path.join(base_dir, f"view_{idx:02d}.png")
    return candidate if os.path.exists(candidate) else None

def _extract_question(problem_text: str):
    """Extract only the final natural language question.

    Raw prompt format (constant preamble + final question):
        ... long instructions ... The question is as follow: <image>How many cylinders are there?\

    We want to return:
        "How many cylinders are there?"

    Assumptions:
      - The marker phrase 'The question is as follow:' (sometimes people write 'as follows:') appears
        exactly once near the end. We defensively take the *last* occurrence in case of noise.
      - An optional '<image>' token may appear immediately after the marker (keep removing all leading occurrences).
      - Trailing backslashes (dataset line continuations) should be stripped.
    """
    if not problem_text:
        return ""

    text = problem_text.strip()
    # Find last occurrence of the marker (allow optional 's' in 'follows')
    marker_regex = re.compile(r'The question is as follow[s]?\s*:\s*', re.IGNORECASE)
    last_match = None
    for m in marker_regex.finditer(text):
        last_match = m
    if last_match is None:
        # Fallback: return stripped text (no marker found)
        return text

    question_part = text[last_match.end():].strip()
    # Remove any leading <image> tokens (could appear multiple times theoretically)
    question_part = re.sub(r'^(?:<image>\s*)+', '', question_part, flags=re.IGNORECASE)
    # Remove trailing literal "\\n" sequences, stray backslashes, or quotes
    question_part = re.sub(r'(?:\\n)+$', '', question_part)  # remove one or more literal \n at end
    question_part = question_part.rstrip('\\').strip().strip('"').strip()
    return question_part

def _question_side(question: str):
    if not question:
        return None
    q = question.lower()
    if re.search(r"\bleft\b", q):
        return 'left'
    if re.search(r"\bright\b", q):
        return 'right'
    if any(tok in q for tok in ['왼쪽', '좌측', '왼 편', '왼', '좌']):
        return 'left'
    if any(tok in q for tok in ['오른쪽', '우측', '오른 편', '오른', '우']):
        return 'right'
    return None

@torch.no_grad()
def _verifier_answer(images: list[str], question: str):
    model, processor = initialize_verifier()
    if model is None or processor is None:
        return None  # 로딩 실패 시 None 반환
    try:
        from PIL import Image
        pil_images = [Image.open(p).convert('RGB') for p in images]
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
        out = processor.batch_decode(gen, skip_special_tokens=True)[0]
        # <answer> 블록 있으면 사용, 없으면 마지막 줄 추출
        ans_blocks = re.findall(r'<answer>(.*?)</answer>', out, re.DOTALL | re.IGNORECASE)
        if ans_blocks:
            ans = ans_blocks[-1].strip()
        else:
            ans = out.strip().split('\n')[-1].strip()
        # 후처리: 끝 구두점 제거
        ans = ans.rstrip('.').strip()
        return ans
    except Exception as e:
        print(f"[action_accuracy] verifier inference failed: {e}")
        return None

def action_accuracy_reward(completions, solution, **kwargs):
    """학생 출력이 선택한 액션(A-L)에 따라 추가 뷰를 선택하고, 해당 뷰(1 또는 2장)를 검증용
    Qwen2.5-VL-7B 모델에 넣어 답을 재추론한 후 GT 와 비교하여 accuracy 점수를 부여.

    Reward = default_accuracy_reward(verifier_answer, ground_truth)
    - verifier 또는 이미지 로딩 실패 시 0.0 (보수적)
    - 두 번째 뷰 파일이 없으면 단일 뷰로 진행
    """
    rewards = []
    contents = [c[0]["content"] for c in completions]
    # image_path: list[list[str]] 혹은 list[str]
    image_paths_arg = kwargs.get("image_path")
    problems_arg = kwargs.get("problem")
    for i, (content, sol) in enumerate(zip(contents, solution)):
        # 원본 이미지 경로 확보
        if isinstance(image_paths_arg, list) and len(image_paths_arg) == len(contents):
            orig_imgs = image_paths_arg[i]
        else:
            # 배치 구조 불명확 시 첫 요소만 활용
            orig_imgs = image_paths_arg[0] if image_paths_arg else []
        if isinstance(orig_imgs, str):
            orig_imgs = [orig_imgs]
        if not orig_imgs:
            rewards.append(0.0)
            continue
        primary_img = orig_imgs[0]
        letter = _extract_final_answer_letter(content)
        # Build dynamic mapping using question side + optional gt_action from dataset item (if present)
        # Expect gt_action could be provided via kwargs per-item in a list under key 'gt_action'
        if isinstance(problems_arg, list) and len(problems_arg) == len(contents):
            raw_problem = problems_arg[i]
        else:
            raw_problem = problems_arg[0] if problems_arg else ''
        question = _extract_question(raw_problem)
        side = _question_side(question)
        mapping = ACTION_LETTERS
        gt_actions_arg = kwargs.get('gt_action')
        gt_action_item = None
        if isinstance(gt_actions_arg, list) and len(gt_actions_arg) == len(contents):
            gt_action_item = gt_actions_arg[i]
        elif isinstance(gt_actions_arg, list) and len(gt_actions_arg) > 0:
            gt_action_item = gt_actions_arg[0]
        if isinstance(gt_action_item, str):
            ga = gt_action_item.strip().upper()
            if ga in ('A', 'B') and side in ('left', 'right'):
                if (side == 'left' and ga == 'B') or (side == 'right' and ga == 'A'):
                    mapping = {'A': 2, 'B': 1}
        # Debug mapping selection
        try:
            if mapping is ACTION_LETTERS:
                map_note = 'normal'
            else:
                map_note = 'inverted'
            print(f"[action_accuracy][map] side={side} gt_action={gt_action_item} mapping={map_note}")
        except Exception:
            pass
        add_view = _build_additional_view(primary_img, letter, mapping)
        selected_images = [primary_img] + ([add_view] if add_view else [])
        verifier_ans = _verifier_answer(selected_images, question)
        if verifier_ans is None:
            reward = 0.0
        else:
            reward = default_accuracy_reward(verifier_ans, sol)
        rewards.append(reward)
        # Always log concise info each train step for debugging selected views
        try:
            sel_indices = [(_view_index_from_path(p)) for p in selected_images]
            print(f"[action_accuracy][debug] sel_indices={sel_indices} question='{question[:80]}' verifier_ans='{verifier_ans}' gt='{sol[:80]}' reward={reward:.3f}")
        except Exception:
            pass
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "debug_action_accuracy.txt")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace('.txt','_action_accuracy.txt'), 'a', encoding='utf-8') as f:
                f.write(f"----- {current_time} action_accuracy reward: {reward} -----\n")
                f.write(f"letter: {letter}, add_view: {add_view}\n")
                f.write(f"selected_images: {selected_images}\n")
                f.write(f"selected_view_indices: {sel_indices}\n")
                f.write(f"question: {question}\n")
                f.write(f"verifier_ans: {verifier_ans}\n")
                f.write(f"ground_truth: {sol}\n")
                f.write(f"student_raw: {content}\n")
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": cosine_rewards,
    "repetition": repetition_rewards,
    "action_accuracy": action_accuracy_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

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
