"""Quick manual test for `_extract_question`.

Run:
  PYTHONPATH=./VLM-R1/src/open-r1-multimodal/src python VLM-R1/src/open-r1-multimodal/tests/test_extract_question.py

Adjust the PYTHONPATH relative to your workspace root if needed.
"""

from open_r1.grpo_jsonl import _extract_question

CASES = {
    "standard": (
        "...instructions... The question is as follow: <image>How many cylinders are there?\\n",
        "How many cylinders are there?",
    ),
    "follows_variant": (
        "Some intro. The question is as follows: <image>What is the color of the box?",
        "What is the color of the box?",
    ),
    "multiple_markers": (
        "Header. The question is as follow: <image>OLD QUESTION? More text. The question is as follow: <image>Final one?",
        "Final one?",
    ),
    "no_marker": (
        "Direct question without marker?",
        "Direct question without marker?",
    ),
    "extra_images": (
        "The question is as follow: <image><image>Describe the scene.",
        "Describe the scene.",
    ),
    "quoted_and_backslash": (
        "The question is as follow: <image>\"How tall is the tower?\"\\n",
        "How tall is the tower?",
    ),
}


def main():
    all_passed = True
    for name, (raw, expected) in CASES.items():
        got = _extract_question(raw)
        ok = (got == expected)
        all_passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}\n  raw: {raw}\n  got: {got}\n  exp: {expected}\n")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
