# GRPO Training Dataset Generation

이 폴더는 CLEVR 스타일로 생성된 새로운 3D 환경 질의응답 데이터를 GRPO 학습 형식(jsonl)으로 변환한 결과를 포함합니다.

## 생성 스크립트
`create_grpo_training_jsonl.py` 는 아래 입력을 사용하여 `training_grpo.jsonl` 파일을 만듭니다.

- 질문 파일(JSON 리스트):
  `/home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/training_questions.json`
- 이미지 경로 템플릿:
  `/home/juil/docker_home/projects/cvpr_2026/clevr-dataset-gen/output/training/scene_{scene_idx:06d}/view_00.png`

## 출력 포맷 (jsonl 한 줄 예시)
```json
{
  "id": 1,
  "image": "/abs/path/scene_000000/view_00.png",
  "conversations": [
    {"from": "human", "value": "<image>...prompt..."},
    {"from": "gpt", "value": "정답"}
  ]
}
```

`human` value에는 다음이 결합됩니다:
1. 카메라 회전/액션 선택 지침(A–L)
2. 실제 질문 문장 (형식: `The question is as follow: <image>{question}`)

## 실행 방법
워크스페이스 Python 환경에서:
```bash
python dataset/create_grpo_training_jsonl.py
```
성공 시 `dataset/training_grpo.jsonl` 가 생성되며 표준출력에 생성 라인 수를 표시합니다.

## 주의 사항
- 원본 질문 JSON 경로가 로컬 환경과 다르면 스크립트 상단 상수를 수정하세요.
- 이미지가 존재하지 않아도 라인 생성은 되므로, 학습 전 경로 유효성 점검을 권장합니다.
- UTF-8 (ensure_ascii=False) 로 저장되어 비라틴 문자가 그대로 유지됩니다.

## 라이선스 / 사용
실험 내부용. 필요 시 별도 라이선스 명시 추가.
