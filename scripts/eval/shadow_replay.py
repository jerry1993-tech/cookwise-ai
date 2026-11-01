#!/usr/bin/env python3
"""
影子回放：把黄金集通过 HTTP 打进运行中的 /api/chat，不改线上流量。

每轮请求后从路由决策 JSONL 里取回真实决策（query、route_key、证据状态、耗时），
与黄金集标注对比，结果逐条写入 shadow_turns.jsonl，汇总写入 shadow_report.json。

前置：
    # 后端已启动，且路由落盘开启（默认开启）
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py

用法：
    python scripts/eval/shadow_replay.py --limit 20
    python scripts/eval/shadow_replay.py --base-url http://localhost:8001
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


class RouteLogTail:
    """增量读取路由 JSONL，按 session_id + query 找本轮决策。"""

    def __init__(self, path: Path):
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0

    def read_new(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as handle:
            handle.seek(self.offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.offset = handle.tell()
        return records

    def find(self, session_id: str, query: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """日志是追加写，请求返回后最多等 3 秒。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for record in self.read_new():
                if record.get("session_id") == session_id and record.get("query") == query:
                    return record
            time.sleep(0.1)
        return None


def predicted_keys(decision: Dict[str, Any]) -> List[str]:
    """澄清轮没有子查询；其余按 sub_queries 统计路由键。"""
    if decision.get("response_type") == "clarify":
        return []
    return [
        str(item.get("route_key") or "")
        for item in (decision.get("sub_queries") or [])
        if item.get("route_key")
    ]


def replay_case(
    base_url: str,
    case: Dict[str, Any],
    tail: RouteLogTail,
    timeout: float,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    session_id = f"shadow-{case['id']}-{int(time.time() * 1000)}"
    turns: List[Dict[str, Any]] = []

    for index, turn in enumerate(case.get("turns") or []):
        query = str(turn.get("query") or "")
        expected = turn.get("expected") or {}
        started = time.perf_counter()
        error = ""
        answer = ""
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={"message": query, "session_id": session_id},
                headers=headers or {},
                timeout=timeout,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
            else:
                payload = response.json()
                answer = str(payload.get("response") or "")
        except requests.RequestException as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            error = f"request: {exc}"

        decision = tail.find(session_id, query) or {}
        keys = predicted_keys(decision) if decision else []
        expected_keys = list(expected.get("route_keys") or [])
        # 答案忠实度的粗代理：期望证据里的关键词/菜名至少有一个出现在回答里
        expected_terms = []
        for item in expected.get("evidence") or []:
            expected_terms.extend(item.get("keywords") or [])
            expected_terms.extend(item.get("recipe_names") or [])
        answer_hit = (
            any(term in answer for term in expected_terms) if expected_terms and answer else None
        )
        turns.append({
            "case_id": case["id"],
            "category": case.get("category"),
            "boundary": bool(case.get("boundary")),
            "turn_index": index,
            "query": query,
            "expected": expected,
            "predicted": {
                "route_keys": keys,
                "route_key": decision.get("route_key", ""),
                "response_type": decision.get("response_type", ""),
                "needs_retrieval": decision.get("needs_retrieval"),
                "confidence": decision.get("confidence"),
                "evidence_status": decision.get("evidence_status", ""),
                "sub_evidence": decision.get("sub_evidence", []),
                "doc_count": decision.get("doc_count", 0),
                "server_latency_ms": decision.get("latency_ms"),
            },
            "route_exact": Counter(expected_keys) == Counter(keys),
            "needs_exact": bool(expected.get("needs_retrieval")) == bool(
                decision.get("needs_retrieval")
            ),
            "response_type_exact": expected.get("response_type") == decision.get("response_type"),
            "answer_keyword_hit": answer_hit,
            "latency_ms": latency_ms,
            "answer_chars": len(answer),
            "error": error,
            "logged": bool(decision),
        })

    return turns


def summarize(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(turns) or 1
    errors = [item for item in turns if item["error"] or not item["logged"]]
    evidence = Counter(item["predicted"]["evidence_status"] or "unknown" for item in turns)
    latencies = [item["latency_ms"] for item in turns]
    server_latencies = [
        item["predicted"]["server_latency_ms"]
        for item in turns
        if item["predicted"]["server_latency_ms"] is not None
    ]
    answer_checked = [item for item in turns if item["answer_keyword_hit"] is not None]
    return {
        "turns": len(turns),
        "errors": len(errors),
        "error_rate": round(len(errors) / total, 4),
        "empty_answer_rate": round(
            sum(1 for item in turns if item["answer_chars"] == 0 and not item["error"]) / total,
            4,
        ),
        "route_key_accuracy": round(
            sum(1 for item in turns if item["route_exact"]) / total, 4
        ),
        "needs_retrieval_accuracy": round(
            sum(1 for item in turns if item["needs_exact"]) / total, 4
        ),
        "response_type_accuracy": round(
            sum(1 for item in turns if item["response_type_exact"]) / total, 4
        ),
        "evidence_status": dict(evidence),
        "evidence_missing_rate": round(evidence.get("missing", 0) / total, 4),
        "evidence_partial_rate": round(evidence.get("partial", 0) / total, 4),
        "answer_keyword_hit_rate": round(
            sum(1 for item in answer_checked if item["answer_keyword_hit"])
            / len(answer_checked),
            4,
        ) if answer_checked else None,
        "answer_checked_turns": len(answer_checked),
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "server_latency_ms_p95": _percentile(server_latencies, 0.95),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--gold", default=str(ROOT / "tests/gold/golden_set.jsonl"))
    parser.add_argument("--route-log", default=str(ROOT / "data/logs/route_decisions.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "tests/gold/shadow_turns.jsonl"))
    parser.add_argument("--report", default=str(ROOT / "tests/gold/shadow_report.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="按固定间隔抽样 N 条，覆盖不同分类；基线和 canary 用同一个 N 可对齐用例",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="额外请求头，格式 'Name: Value'，可重复；canary 用 --header 'X-Canary: 1'",
    )
    args = parser.parse_args()

    headers = {}
    for item in args.header:
        name, _, value = item.partition(":")
        if name.strip():
            headers[name.strip()] = value.strip()

    cases: List[Dict[str, Any]] = []
    with Path(args.gold).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[: args.limit]
    if args.sample and args.sample < len(cases):
        step = len(cases) / args.sample
        cases = [cases[int(index * step)] for index in range(args.sample)]
    if not cases:
        print("黄金集为空", file=sys.stderr)
        return 1

    tail = RouteLogTail(Path(args.route_log))
    started = time.time()
    turns: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        turns.extend(replay_case(args.base_url, case, tail, args.timeout, headers))
        if index % 10 == 0 or index == len(cases):
            print(f"  已回放 {index}/{len(cases)} 条", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in turns:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    summary = summarize(turns)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["base_url"] = args.base_url
    summary["gold"] = args.gold
    report_path = Path(args.report)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 影子回放 ===")
    for key, value in summary.items():
        print(f"{key:28s} {value}")
    print(f"\n逐轮明细 -> {out_path}")
    print(f"汇总 -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
