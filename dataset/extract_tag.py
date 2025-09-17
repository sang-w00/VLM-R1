import json, re, argparse, shutil, os
from typing import Optional

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

def extract_answer(raw: str) -> str:
    """
    <answer> ... </answer> 안의 내용을 추출하고 숫자만 남김.
    숫자가 하나도 없으면 태그 내부 원본을 깔끔히 정리해 반환.
    """
    if not isinstance(raw, str):
        return str(raw)
    m = ANSWER_RE.search(raw)
    if m:
        inner = m.group(1).strip()
    else:
        inner = raw.strip()
    # 숫자만 남기기 (정수/소수 모두 허용하려면 아래 로직 확장 가능)
    digits = re.findall(r"[0-9]+(?:\\.[0-9]+)?", inner)
    if len(digits) == 1:
        return digits[0]
    elif len(digits) > 1:
        # 여러 숫자가 있으면 공백 구분으로 합침 (원하면 첫 번째만 쓰도록 변경 가능)
        return " ".join(digits)
    else:
        # 숫자가 전혀 없으면 그냥 태그 제거 결과 반환
        return inner

def process_file(inp: str, out: Optional[str], in_place: bool):
    if in_place and out:
        raise ValueError("in-place 모드에서는 out 경로를 지정하지 마세요.")
    if in_place:
        backup = inp + ".bak"
        shutil.copy2(inp, backup)
        print(f"[INFO] 백업 생성: {backup}")
        tmp_out = inp + ".tmp"
        target_out = inp
    else:
        tmp_out = out + ".tmp"
        target_out = out

    total = changed = 0
    with open(inp, "r", encoding="utf-8") as fin, open(tmp_out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON 파싱 실패, 그대로 통과: {line[:80]}...")
                fout.write(line + "\n")
                continue

            convs = obj.get("conversations")
            if isinstance(convs, list):
                for c in convs:
                    if isinstance(c, dict) and c.get("from") == "gpt" and "value" in c:
                        orig = c["value"]
                        new_val = extract_answer(orig)
                        if new_val != orig:
                            c["value"] = new_val
                            changed += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(tmp_out, target_out)
    print(f"[DONE] 총 {total} 줄 처리, 수정 {changed}항목")
    if in_place:
        print(f"[INFO] 원본 덮어씀: {target_out} (백업: {backup})")
    else:
        print(f"[INFO] 출력 파일: {target_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL conversations gpt answer 태그 제거 및 숫자만 남기기")
    parser.add_argument("--input", required=True, help="입력 JSONL 경로")
    parser.add_argument("--output", help="출력 JSONL (in-place 사용 시 지정 금지)")
    parser.add_argument("--in-place", action="store_true", help="제자리(in-place) 수정 (백업 .bak 생성)")
    args = parser.parse_args()
    process_file(args.input, args.output, args.in_place)