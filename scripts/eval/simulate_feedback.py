#!/usr/bin/env python3
"""
模拟用户反馈：给 /api/feedback 打点赞、点踩和纠错，验证落盘与统计。

用法：
    python scripts/eval/simulate_feedback.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--gold", default=str(ROOT / "tests/gold/golden_set.jsonl"))
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    cases = []
    with Path(args.gold).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    cases = cases[: args.limit]
    if not cases:
        print("黄金集为空", file=sys.stderr)
        return 1

    base = args.base_url.rstrip("/")
    for index, case in enumerate(cases):
        query = case["turns"][0]["query"]
        rating = "like" if index % 2 == 0 else "dislike"
        payload = {
            "session_id": f"feedback-sim-{case['id']}",
            "query": query,
            "answer": "（模拟答案）",
            "rating": rating,
            "route_key": "",
        }
        response = requests.post(f"{base}/api/feedback", json=payload, timeout=10)
        print(index + 1, rating, response.status_code, response.text[:120])

    # 再补一条纠错
    correction = {
        "session_id": "feedback-sim-correction",
        "query": cases[0]["turns"][0]["query"],
        "answer": "（模拟答案）",
        "correction": "这里应该先回答食材，再补充步骤。",
    }
    response = requests.post(f"{base}/api/feedback", json=correction, timeout=10)
    print("correction", response.status_code, response.text[:120])

    stats = requests.get(f"{base}/api/feedback/stats", timeout=10)
    print("stats", stats.status_code, json.dumps(stats.json(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
