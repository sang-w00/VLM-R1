from datasets import load_dataset, load_from_disk, Image
from pathlib import Path
import shutil
import json
from tqdm import tqdm
import os

# ===== 경로 설정 =====
ROOT = Path("clevr_cogen_a_train")        # 최상위 폴더
IMG_DIR = ROOT / "data" / "images"           # 샘플과 동일한 하위 경로
OUT_JSONL = ROOT / "clevr_cogen_a_train.jsonl"

IMG_DIR.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)

# ===== 데이터셋 로드 =====
# 1) 환경변수 LOCAL_DATASET_DIR 가 지정되면 해당 로컬 경로(예: HF 캐시 경로)를 load_from_disk 로 사용
# 2) 아니면 load_dataset 으로 정상 로드 (이미 캐시에 있으면 네트워크 호출 없이 재사용)

LOCAL_DATASET_DIR = os.environ.get("LOCAL_DATASET_DIR", "").strip()

if LOCAL_DATASET_DIR:
    print(f"[INFO] Using local dataset directory: {LOCAL_DATASET_DIR}")
    try:
        loaded = load_from_disk(LOCAL_DATASET_DIR)
        # load_from_disk 결과가 dict(splits) 형태일 수 있음
        if isinstance(loaded, dict):
            if 'train' in loaded:
                ds = loaded['train']
            else:
                raise ValueError("로컬 경로에서 'train' split 을 찾을 수 없습니다.")
        else:
            ds = loaded
    except Exception as e:
        print(f"[WARN] load_from_disk 실패: {e}\n[INFO] arrow shard 직접 로드 시도")
        import glob
        from datasets import Dataset, concatenate_datasets

        # 1) 지정된 디렉토리 바로 아래에서 arrow 파일 수집
        arrow_pattern = str(Path(LOCAL_DATASET_DIR) / "*.arrow")
        shard_paths = sorted(glob.glob(arrow_pattern))

        # 2) 없으면 하위 'train' 폴더 탐색
        if not shard_paths:
            arrow_pattern = str(Path(LOCAL_DATASET_DIR) / "train" / "*.arrow")
            shard_paths = sorted(glob.glob(arrow_pattern))

        # 3) clevr_cogen 관련 prefix 필터링(임시 tmp* 파일 제외)
        filtered = [p for p in shard_paths if 'clevr_cogen_a_train-train-' in Path(p).name and not Path(p).name.startswith('tmp')]
        if filtered:
            shard_paths = filtered

        if not shard_paths:
            raise FileNotFoundError(f"arrow shard 파일을 찾을 수 없습니다: {LOCAL_DATASET_DIR}")

        print(f"[INFO] 발견된 shard 수: {len(shard_paths)} (예: {shard_paths[0]})")
        parts = [Dataset.from_file(p) for p in shard_paths]
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        print(f"[INFO] 수동 결합 완료: {len(ds)} rows")
else:
    print("[INFO] LOCAL_DATASET_DIR 미지정 -> load_dataset 사용")
    ds = load_dataset("leonardPKU/clevr_cogen_a_train", split="train")

# decode=False 로 캐시 내 실제 파일 path 확보
ds = ds.cast_column("image", Image(decode=False))

# 문제/정답 컬럼 이름 유연 처리
def get_problem(ex):
    for key in ["problem", "question", "prompt"]:
        if key in ex:
            return ex[key]
    raise KeyError("문항 텍스트 컬럼을 찾을 수 없습니다. (problem/question/prompt) 중 하나가 필요")

def get_solution(ex):
    for key in ["solution", "answer", "label"]:
        if key in ex:
            return ex[key]
    raise KeyError("정답 컬럼을 찾을 수 없습니다. (solution/answer/label) 중 하나가 필요")

# ===== 내보내기 =====
with OUT_JSONL.open("w", encoding="utf-8") as fout:
    for idx, ex in tqdm(enumerate(ds, start=1), total=len(ds), desc="Exporting"):
        # 원본 파일명 얻기 (HF 캐시 경로에서 basename만 추출)
        src_info = ex["image"]
        if isinstance(src_info, dict) and "path" in src_info and src_info["path"]:
            basename = Path(src_info["path"]).name
        else:
            # 드물게 path가 비어있을 수 있으니 fallback 파일명 지정
            basename = f"CLEVR_trainA_{idx:06d}.png"

        dst_path = IMG_DIR / basename

        # 이미지 파일 복사 (디코딩/재인코딩 없이 그대로 복사)
        if isinstance(src_info, dict) and "path" in src_info and src_info["path"]:
            try:
                shutil.copy2(src_info["path"], dst_path)
            except Exception:
                # 혹시 복사가 실패하면 decode=True로 다시 로드해서 저장하는 fallback
                # (성능은 떨어지지만 안정성 확보)
                pil_ds = ds.cast_column("image", Image(decode=True))
                pil_img = pil_ds[idx - 1]["image"]   # idx는 1부터 시작이므로 보정
                pil_img.save(dst_path)
        else:
            # path가 없을 때: decode=True로 저장
            pil_ds = ds.cast_column("image", Image(decode=True))
            pil_img = pil_ds[idx - 1]["image"]
            pil_img.save(dst_path)

        # 문항/정답
        problem_text = str(get_problem(ex))
        solution_text = str(get_solution(ex))  # 샘플처럼 문자열로 기록

        # 요청 스키마에 맞춘 레코드 작성
        record = {
            "id": idx,  # 1부터 순번
            "image": str((ROOT / "data" / "images" / basename).as_posix()),
            "conversations": [
                {"from": "human", "value": f"<image>{problem_text}"},
                {"from": "gpt", "value": solution_text},
            ],
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"[OK] Wrote JSONL to {OUT_JSONL} and images to {IMG_DIR}")
