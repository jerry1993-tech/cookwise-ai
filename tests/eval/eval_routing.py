#!/usr/bin/env python3
"""
跑黄金集，评估路由 / 检索决策 / 证据命中。

默认在进程内初始化完整系统（Neo4j + Milvus + LLM），每条 case 用独立 session，
按生产路径调用 ``route_query`` 和 ``update_state_after_turn``。

用法：
    # 先抽 20 条看看
    python tests/eval/eval_routing.py --limit 20

    # 全量，4 个 case 并发
    python tests/eval/eval_routing.py --workers 4

输出：
    tests/gold/eval_report.json
    tests/gold/predictions.jsonl   （给阈值校准用）
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
# 本机模型已缓存：强制离线，避免评测启动时卡在 huggingface.co 超时
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from rag_modules.query_routing.domain import NO_RETRIEVAL_KEYS, ROUTE_SPECS

ALL_KEYS = list(ROUTE_SPECS)
PRINT_LOCK = threading.Lock()


def _percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _normalize_slot(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "").strip()


def _doc_route_keys(doc) -> set:
    metadata = getattr(doc, "metadata", {}) or {}
    return {
        str(metadata.get("sub_route_key") or ""),
        str(metadata.get("route_key") or ""),
    } - {""}


def _evidence_hit(item: Dict[str, Any], documents: List[Any]) -> Tuple[bool, List[str]]:
    """一条证据是否被同一份文档同时覆盖；返回失败时的文档摘要。"""
    route_key = str(item.get("route_key") or "")
    keywords = [str(word) for word in item.get("keywords") or []]
    recipe_names = [str(name) for name in item.get("recipe_names") or []]
    forbidden = [str(word) for word in item.get("forbidden") or []]
    match = item.get("match", "all")

    candidates = [doc for doc in documents if route_key in _doc_route_keys(doc)]
    snippets: List[str] = []
    for doc in candidates:
        content = str(getattr(doc, "page_content", "") or "")
        metadata = getattr(doc, "metadata", {}) or {}
        snippets.append(content[:200].replace("\n", " "))
        if any(word and word in content for word in forbidden):
            continue
        names_ok = all(
            name in content or str(metadata.get("recipe_name") or "") == name
            for name in recipe_names
        )
        if not names_ok:
            continue
        if match == "any":
            words_ok = not keywords or any(word in content for word in keywords)
        else:
            words_ok = all(word in content for word in keywords)
        if words_ok:
            return True, snippets
    return False, snippets[:5]


def _route_exact(expected: List[str], predicted: List[str]) -> bool:
    return Counter(expected) == Counter(predicted)


def _slots_ok(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    for key, value in expected.items():
        if _normalize_slot(actual.get(key)) != _normalize_slot(value):
            return False
    return True


def evaluate_case(system, case: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    """按生产路径跑完一条 case，返回每个 turn 的预测与判定。"""
    session_id = f"gold-{case['id']}-{int(time.time() * 1000)}"
    records: List[Dict[str, Any]] = []

    for index, turn in enumerate(case.get("turns") or []):
        query = str(turn.get("query") or "")
        expected = turn.get("expected") or {}
        state = system.cache_manager.get_state(session_id)

        started = time.perf_counter()
        documents, analysis = system.query_router.route_query(
            query,
            top_k,
            session_id=session_id,
            session_state=state,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        system.cache_manager.update_state_after_turn(session_id, query, analysis, "")

        predicted_keys = [
            str(item.get("route_key") or "")
            for item in (getattr(analysis, "sub_queries", None) or [])
        ]
        if getattr(analysis, "response_type", "") == "clarify":
            predicted_keys = []

        expected_keys = list(expected.get("route_keys") or [])
        expected_evidence = list(expected.get("evidence") or [])
        evidence_flags = []
        evidence_debug = []
        for item in expected_evidence:
            hit, snippets = _evidence_hit(item, documents)
            evidence_flags.append(hit)
            if not hit:
                evidence_debug.append({
                    "route_key": item.get("route_key"),
                    "keywords": item.get("keywords"),
                    "recipe_names": item.get("recipe_names"),
                    "docs": snippets,
                })

        evidence_hit = all(evidence_flags) if evidence_flags else None
        expected_slots = expected.get("slots") or {}
        # 多意图时 analysis.slots 只是主意图的槽位，不拿它判多意图槽位
        slots_ok = None
        if expected_slots and len(expected_keys) == 1:
            slots_ok = _slots_ok(expected_slots, getattr(analysis, "slots", None) or {})

        records.append({
            "case_id": case["id"],
            "category": case.get("category"),
            "boundary": bool(case.get("boundary")),
            "multi_intent": bool(case.get("multi_intent")),
            "multi_turn": bool(case.get("multi_turn")),
            "turn_index": index,
            "query": query,
            "expected": expected,
            "predicted": {
                "route_keys": predicted_keys,
                "route_key": getattr(analysis, "route_key", ""),
                "intent": getattr(analysis, "intent", ""),
                "needs_retrieval": bool(getattr(analysis, "needs_retrieval", False)),
                "response_type": getattr(analysis, "response_type", ""),
                "confidence": float(getattr(analysis, "confidence", 0.0) or 0.0),
                "slots": getattr(analysis, "slots", {}) or {},
                "is_multi_intent": bool(getattr(analysis, "is_multi_intent", False)),
                "evidence_status": getattr(analysis, "evidence_status", ""),
                "sub_evidence": getattr(analysis, "sub_evidence", []) or [],
                "sub_queries": getattr(analysis, "sub_queries", []) or [],
                "understanding_source": (getattr(analysis, "understanding", None) or {}).get("source", ""),
                "understanding": getattr(analysis, "understanding", None) or {},
                "doc_count": len(documents),
                "latency_ms": latency_ms,
            },
            "route_exact": _route_exact(expected_keys, predicted_keys),
            "needs_exact": bool(expected.get("needs_retrieval")) == bool(
                getattr(analysis, "needs_retrieval", False)
            ),
            "response_exact": expected.get("response_type") == getattr(
                analysis, "response_type", ""
            ),
            "evidence_hit": evidence_hit,
            "evidence_debug": evidence_debug,
            "slots_ok": slots_ok,
        })

    return records


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    turns = len(records)
    if not turns:
        return {}

    route_correct = sum(1 for item in records if item["route_exact"])
    needs_correct = sum(1 for item in records if item["needs_exact"])
    response_correct = sum(1 for item in records if item["response_exact"])
    evidence_records = [item for item in records if item["evidence_hit"] is not None]
    evidence_correct = sum(1 for item in evidence_records if item["evidence_hit"])
    slot_records = [item for item in records if item["slots_ok"] is not None]
    slot_correct = sum(1 for item in slot_records if item["slots_ok"])

    expected_clarify = sum(
        1 for item in records if (item["expected"].get("response_type") == "clarify")
    )
    predicted_clarify = sum(
        1 for item in records if item["predicted"].get("response_type") == "clarify"
    )
    false_clarify = sum(
        1 for item in records
        if item["predicted"].get("response_type") == "clarify"
        and item["expected"].get("response_type") != "clarify"
    )
    missed_clarify = sum(
        1 for item in records
        if item["expected"].get("response_type") == "clarify"
        and item["predicted"].get("response_type") != "clarify"
    )

    key_tp = key_fp = key_fn = 0
    per_key: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"support": 0, "tp": 0, "fp": 0, "fn": 0}
    )
    for item in records:
        expected_keys = item["expected"].get("route_keys") or []
        predicted_keys = item["predicted"].get("route_keys") or []
        expected_counter = Counter(expected_keys)
        predicted_counter = Counter(predicted_keys)
        for key, count in expected_counter.items():
            per_key[key]["support"] += count
            matched = min(count, predicted_counter.get(key, 0))
            per_key[key]["tp"] += matched
            per_key[key]["fn"] += count - matched
            key_tp += matched
            key_fn += count - matched
        for key, count in predicted_counter.items():
            extra = count - expected_counter.get(key, 0)
            if extra > 0:
                per_key[key]["fp"] += extra
                key_fp += extra

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted({item["category"] for item in records}):
        subset = [item for item in records if item["category"] == category]
        subset_evidence = [item for item in subset if item["evidence_hit"] is not None]
        by_category[category] = {
            "turns": len(subset),
            "route_accuracy": round(sum(1 for item in subset if item["route_exact"]) / len(subset), 4),
            "needs_accuracy": round(sum(1 for item in subset if item["needs_exact"]) / len(subset), 4),
            "evidence_hit_rate": round(
                sum(1 for item in subset_evidence if item["evidence_hit"]) / len(subset_evidence), 4
            ) if subset_evidence else None,
        }

    latencies = [item["predicted"]["latency_ms"] for item in records]
    sources = Counter(item["predicted"].get("understanding_source") or "unknown" for item in records)

    return {
        "turns": turns,
        "cases": len({(item["case_id"]) for item in records}),
        "route_key_accuracy": round(route_correct / turns, 4),
        "route_key_micro_precision": round(key_tp / (key_tp + key_fp), 4) if key_tp + key_fp else None,
        "route_key_micro_recall": round(key_tp / (key_tp + key_fn), 4) if key_tp + key_fn else None,
        "needs_retrieval_accuracy": round(needs_correct / turns, 4),
        "response_type_accuracy": round(response_correct / turns, 4),
        "evidence_hit_rate": round(evidence_correct / len(evidence_records), 4) if evidence_records else None,
        "evidence_checked_turns": len(evidence_records),
        "slot_accuracy": round(slot_correct / len(slot_records), 4) if slot_records else None,
        "slot_checked_turns": len(slot_records),
        "clarify_rate": round(predicted_clarify / turns, 4),
        "expected_clarify_rate": round(expected_clarify / turns, 4),
        "false_clarify": false_clarify,
        "missed_clarify": missed_clarify,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "understanding_sources": dict(sources),
        "by_category": by_category,
        "per_route_key": {
            key: {
                **values,
                "precision": round(values["tp"] / (values["tp"] + values["fp"]), 4)
                if values["tp"] + values["fp"] else None,
                "recall": round(values["tp"] / (values["tp"] + values["fn"]), 4)
                if values["tp"] + values["fn"] else None,
            }
            for key, values in sorted(per_key.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(ROOT / "tests/gold/golden_set.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "tests/gold/eval_report.json"))
    parser.add_argument("--predictions", default=str(ROOT / "tests/gold/predictions.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    os.chdir(ROOT)
    cases: List[Dict[str, Any]] = []
    with Path(args.gold).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("黄金集为空", file=sys.stderr)
        return 1

    from config import DEFAULT_CONFIG
    from main import AdvancedGraphRAGSystem

    print(f"初始化系统，准备跑 {len(cases)} 条 case...", flush=True)
    system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
    system.initialize_system()
    system.build_knowledge_base()
    print("系统就绪", flush=True)

    records: List[Dict[str, Any]] = []
    completed = 0
    started = time.time()

    if args.workers <= 1:
        for case in cases:
            records.extend(evaluate_case(system, case, DEFAULT_CONFIG.top_k))
            completed += 1
            if completed % 10 == 0 or completed == len(cases):
                print(f"  已完成 {completed}/{len(cases)} 条", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(evaluate_case, system, case, DEFAULT_CONFIG.top_k): case
                for case in cases
            }
            for future in as_completed(futures):
                case = futures[future]
                try:
                    records.extend(future.result())
                except Exception as exc:
                    print(f"case 失败 {case['id']}: {exc}", flush=True)
                completed += 1
                if completed % 10 == 0 or completed == len(cases):
                    with PRINT_LOCK:
                        print(f"  已完成 {completed}/{len(cases)} 条", flush=True)

    summary = _summarize(records)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["gold"] = str(args.gold)
    summary["cases_requested"] = len(cases)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_path = Path(args.predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    print("\n=== 总体 ===")
    for key in (
        "turns", "route_key_accuracy", "route_key_micro_precision",
        "route_key_micro_recall", "needs_retrieval_accuracy",
        "response_type_accuracy", "evidence_hit_rate", "slot_accuracy",
        "clarify_rate", "expected_clarify_rate", "false_clarify", "missed_clarify",
        "latency_ms_p50", "latency_ms_p95", "elapsed_seconds",
    ):
        print(f"{key:28s} {summary.get(key)}")
    print("\n=== 分类 ===")
    for category, values in summary["by_category"].items():
        print(f"{category:30s} {values}")
    print(f"\n报告 -> {out_path}")
    print(f"预测 -> {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
