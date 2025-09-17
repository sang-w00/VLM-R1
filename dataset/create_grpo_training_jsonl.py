import json
import os
from pathlib import Path
from typing import List, Dict

"""Generate GRPO training jsonl from CLEVR-like question file.

Input question file format (JSON list):
[
  {
    "question": "How many objects are there?",
    "answer": "4",
    "question_type": "counting",
    "scene_index": 0
  },
  ...
]

Image path template:
/home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/training/scene_{SCENE_IDX:06d}/view_00.png

Output jsonl line example:
{
  "id": 1,
  "image": ".../scene_000000/view_00.png",
  "conversations": [
    {"from": "human", "value": "<image>...prompt..."},
    {"from": "gpt", "value": "..."}
  ]
}
"""

QUESTION_JSON = Path("/home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/test_questions.json")
IMAGE_TEMPLATE = "/home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/test/scene_{scene_idx:06d}/view_00.png"
OUTPUT_JSONL = Path(__file__).parent / "test_grpo.jsonl"

BASE_PROMPT_PREFIX = (
    "You are given an image showing a specific view of a 3D environment. In this scene, the largest object is placed at the center, surrounded by several smaller objects, some of which may be partially occluded from this viewpoint. A total of 12 cameras orbit around the central object, each separated by 30 degrees."
    "Your task is to select one action from the options (A–L) below in order to best answer a given question about this environment:"\
    "Option L — If the current view provides enough information, answer the question directly using only this view.\n"\
    "Options A–K — If the current view is not sufficient, request one additional image by rotating the camera clockwise around the central object at the specified angle.\n"\
    "Actions:\n"\
    "A. Rotate 30° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "B. Rotate 60° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "C. Rotate 90° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "D. Rotate 120° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "E. Rotate 150° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "F. Rotate 180° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "G. Rotate 210° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "H. Rotate 240° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "I. Rotate 270° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "J. Rotate 300° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "K. Rotate 330° clockwise around the central object’s vertical axis and look at the central object again.\n"\
    "L. Do not move; answer using only the current view.\n"
)

QUESTION_PREFIX = "The question is as follow: <image>{question}\n"


def load_questions(path: Path) -> List[Dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Question file must be a list of question objects.")
    required_keys = {"question", "answer", "scene_index"}
    for i, item in enumerate(data):
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Item {i} missing keys: {missing}")
    return data


def build_conversation(question: str) -> str:
    return BASE_PROMPT_PREFIX + QUESTION_PREFIX.format(question=question)


def main():
    questions = load_questions(QUESTION_JSON)
    out_lines = 0
    with open(OUTPUT_JSONL, 'w') as out_f:
        for idx, qa in enumerate(questions, start=1):
            scene_idx = qa["scene_index"]
            image_path = IMAGE_TEMPLATE.format(scene_idx=scene_idx)
            question_text = qa["question"].strip()
            answer_text = qa["answer"].strip()

            human_value = build_conversation(question_text)
            record = {
                "id": idx,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": answer_text}
                ]
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_lines += 1
    print(f"Wrote {out_lines} lines to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
