import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.answer import get_answer

THRESHOLD = 0.5

def simple_score(answer, expected):
    answer_lower = answer.lower()
    keywords = expected.lower().split()
    hits = sum(1 for kw in keywords if kw in answer_lower)
    return hits / len(keywords)

def run_evals(path="eval/golden_qa.json"):
    with open(path) as f:
        dataset = json.load(f)

    results = []
    passed = 0

    for item in dataset:
        print(f"Testing: {item['question']}")
        answer, citations = get_answer(item["question"])
        score = simple_score(answer, item["expected"])
        status = "PASS" if score >= THRESHOLD else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"[{status}] ({score:.2f}) {item['question']}")

    overall = passed / len(dataset)
    print(f"\nResult: {passed}/{len(dataset)} passed ({overall*100:.0f}%)")

    if overall < THRESHOLD:
        print("EVAL FAILED - below threshold")
        sys.exit(1)
    else:
        print("EVAL PASSED")

if __name__ == "__main__":
    run_evals()