#!/usr/bin/env python3
"""
按路由键校准三个置信度阈值：min_execute / retrieval_override / no_retrieval。

输入是 ``eval_routing.py`` 产出的 predictions.jsonl（里面有每次 LLM 理解结果），
不重新调用 LLM，只用生产 RoutePolicy 回放不同阈值下的决策。

- 70% case 做训练，30% case 做留出验证（按 case 切分，多轮不会跨集）
- 坐标下降：逐个路由键在小网格里试，保留训练集上路由/检索判断更好的组合
- 输出可直接给 ROUTE_THRESHOLDS_PATH 使用

用法：
    python tests/eval/calibrate_thresholds.py
    ROUTE_THRESHOLDS_PATH=tests/gold/calibrated_thresholds.json \
        python tests/eval/eval_routing.py
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_modules.query_routing.domain import (
    DOMAIN,
    NO_RETRIEVAL_KEYS,
    ROUTE_SPECS,
    RouteThreshold,
)
from rag_modules.query_routing.policy import RoutePolicy
from rag_modules.query_routing.schemas import (
    ContextInfo,
    EntityRef,
    IntentItem,
    Understanding,
)

ALL_KEYS = list(ROUTE_SPECS)
DEFAULT = RouteThreshold()
MIN_EXECUTE_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)
OVERRIDE_GRID = (0.50, 0.60, 0.70, 0.80)
NO_RETRIEVAL_GRID = (0.70, 0.80, 0.90, 0.95)


def build_understanding(payload: Dict[str, Any]) -> Understanding:
    """把 predictions.jsonl 里保存的理解结果还原成 Understanding。"""
    intents = []
    for item in payload.get("intents") or []:
        intents.append(IntentItem(
            intent=str(item.get("intent") or ""),
            route_key=str(item.get("route_key") or ""),
            evidence_profile=list(item.get("evidence_profile") or []),
            confidence=float(item.get("confidence") or 0.0),
            span=str(item.get("span") or ""),
            entities=[
                EntityRef(
                    name=str(entity.get("name") or ""),
                    type=str(entity.get("type") or "unknown"),
                    resolved_id=str(entity.get("resolved_id") or ""),
                    unresolved=bool(entity.get("unresolved")),
                )
                for entity in item.get("entities") or []
            ],
            slots=dict(item.get("slots") or {}),
            need_complete=bool(item.get("need_complete")),
            value_policy=str(item.get("value_policy") or ""),
            expected_source=list(item.get("expected_source") or []),
            engine_override=str(item.get("engine_override") or ""),
        ))

    context_payload = payload.get("context") or {}
    context = ContextInfo(
        used_history=bool(context_payload.get("used_history")),
        topic_shift=bool(context_payload.get("topic_shift")),
        active_recipe=str(context_payload.get("active_recipe") or ""),
        active_topic=str(context_payload.get("active_topic") or ""),
        resolved_entities=list(context_payload.get("resolved_entities") or []),
    )
    return Understanding(
        standalone_query=str(payload.get("standalone_query") or ""),
        needs_retrieval=bool(payload.get("needs_retrieval")),
        intents=intents,
        context=context,
        is_ambiguous=bool(payload.get("is_ambiguous")),
        clarify_question=str(payload.get("clarify_question") or ""),
        disclaimer_required=bool(payload.get("disclaimer_required")),
        source=str(payload.get("source") or "llm"),
        raw=dict(payload.get("raw") or {}),
    )


def is_train_case(case_id: str, train_ratio: float) -> bool:
    """按 case 做稳定切分，同一 case 的多轮不会分到两边。"""
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000 < int(train_ratio * 1000)


def plan_keys(plan) -> List[str]:
    return [sub_plan.intent.route_key for sub_plan in plan.sub_plans]


def evaluate(
    records: List[Dict[str, Any]],
    understandings: List[Understanding],
    thresholds: Dict[str, RouteThreshold],
    indices: Iterable[int],
) -> Dict[str, int]:
    """在给定阈值下回放，返回用于比较的计数。"""
    domain = replace(DOMAIN, route_thresholds=thresholds)
    policy = RoutePolicy(domain)
    counts = Counter()
    for index in indices:
        record = records[index]
        expected = record["expected"]
        plan = policy.plan(understandings[index], top_k=5)

        expected_keys = Counter(expected.get("route_keys") or [])
        predicted_keys = Counter(plan_keys(plan))
        counts["turns"] += 1
        counts["route_correct"] += expected_keys == predicted_keys
        counts["needs_correct"] += (
            bool(expected.get("needs_retrieval")) == bool(plan.needs_retrieval)
        )
        expected_clarify = expected.get("response_type") == "clarify"
        predicted_clarify = plan.response_type == "clarify"
        counts["expected_clarify"] += expected_clarify
        counts["predicted_clarify"] += predicted_clarify
        counts["false_clarify"] += predicted_clarify and not expected_clarify
        counts["missed_clarify"] += expected_clarify and not predicted_clarify
    return dict(counts)


def score(counts: Dict[str, int]) -> Tuple[int, int, int]:
    """先保路由正确，再保检索判断；澄清越少越好（业务要求）。"""
    return (
        counts["route_correct"],
        counts["needs_correct"],
        -counts["false_clarify"],
    )


def threshold_distance(candidate: RouteThreshold, baseline: RouteThreshold) -> float:
    return (
        abs(candidate.min_execute - baseline.min_execute)
        + abs(candidate.retrieval_override - baseline.retrieval_override)
        + abs(candidate.no_retrieval - baseline.no_retrieval)
    )


def candidate_configs(key: str, baseline: RouteThreshold) -> List[RouteThreshold]:
    """每个键只调真正会影响它的字段，避免无谓漂移。"""
    if key in NO_RETRIEVAL_KEYS:
        # 闲聊/域外键不产生检索计划，min_execute 与 override 不会生效
        return [
            replace(baseline, no_retrieval=value)
            for value in NO_RETRIEVAL_GRID
        ]
    candidates = []
    for min_execute in MIN_EXECUTE_GRID:
        for override in OVERRIDE_GRID:
            if min_execute > override:
                continue
            candidates.append(replace(
                baseline,
                min_execute=min_execute,
                retrieval_override=override,
            ))
    return candidates


def calibrate(
    records: List[Dict[str, Any]],
    understandings: List[Understanding],
    train_indices: List[int],
    passes: int,
) -> Tuple[Dict[str, RouteThreshold], Dict[str, Any]]:
    thresholds = {key: DEFAULT for key in ALL_KEYS}
    baseline_counts = evaluate(records, understandings, thresholds, train_indices)
    history: List[Dict[str, Any]] = []

    for pass_index in range(passes):
        changed = 0
        for key in ALL_KEYS:
            current = thresholds[key]
            current_counts = evaluate(records, understandings, thresholds, train_indices)
            best = current
            best_counts = current_counts
            best_score = score(current_counts)

            for candidate in candidate_configs(key, current):
                trial = dict(thresholds)
                trial[key] = candidate
                counts = evaluate(records, understandings, trial, train_indices)
                candidate_score = score(counts)
                if candidate_score > best_score or (
                    candidate_score == best_score
                    and threshold_distance(candidate, current) < threshold_distance(best, current)
                ):
                    best = candidate
                    best_counts = counts
                    best_score = candidate_score

            if best != current:
                thresholds[key] = best
                changed += 1
                history.append({
                    "pass": pass_index + 1,
                    "key": key,
                    "before": current.__dict__,
                    "after": best.__dict__,
                    "train": best_counts,
                })
        if not changed:
            break

    final_counts = evaluate(records, understandings, thresholds, train_indices)
    # 安全兜底：校准后整体不能比默认差
    if score(final_counts) < score(baseline_counts):
        thresholds = {key: DEFAULT for key in ALL_KEYS}
        final_counts = baseline_counts

    return thresholds, {
        "baseline_train": baseline_counts,
        "calibrated_train": final_counts,
        "changes": history,
    }


def format_thresholds(thresholds: Dict[str, RouteThreshold]) -> Dict[str, Dict[str, float]]:
    payload = {"default": dict(DEFAULT.__dict__)}
    for key in ALL_KEYS:
        value = thresholds[key]
        payload[key] = {
            "min_execute": value.min_execute,
            "retrieval_override": value.retrieval_override,
            "no_retrieval": value.no_retrieval,
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "tests/gold/predictions.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "tests/gold/calibrated_thresholds.json"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "tests/gold/calibration_report.json"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    with Path(args.predictions).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        print("predictions.jsonl 为空，先跑 eval_routing.py", file=sys.stderr)
        return 1

    understandings = [
        build_understanding(record["predicted"]["understanding"])
        for record in records
    ]
    train_indices = [
        index for index, record in enumerate(records)
        if is_train_case(str(record["case_id"]), args.train_ratio)
    ]
    train_set = set(train_indices)
    test_indices = [index for index in range(len(records)) if index not in train_set]

    thresholds, details = calibrate(
        records, understandings, train_indices, args.passes
    )

    default_thresholds = {key: DEFAULT for key in ALL_KEYS}
    train_default = evaluate(records, understandings, default_thresholds, train_indices)
    test_default = evaluate(records, understandings, default_thresholds, test_indices)
    train_calibrated = evaluate(records, understandings, thresholds, train_indices)
    test_calibrated = evaluate(records, understandings, thresholds, test_indices)

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(format_thresholds(thresholds), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "predictions": args.predictions,
        "train_ratio": args.train_ratio,
        "turns_train": len(train_indices),
        "turns_test": len(test_indices),
        "default": {
            "thresholds": format_thresholds(default_thresholds),
            "train": train_default,
            "test": test_default,
        },
        "calibrated": {
            "thresholds": format_thresholds(thresholds),
            "train": train_calibrated,
            "test": test_calibrated,
        },
        "changes": details["changes"],
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def line(label: str, counts: Dict[str, int]) -> str:
        turns = counts["turns"] or 1
        return (
            f"{label:18s} route={counts['route_correct'] / turns:.4f} "
            f"needs={counts['needs_correct'] / turns:.4f} "
            f"clarify={counts['predicted_clarify']} "
            f"false={counts['false_clarify']} missed={counts['missed_clarify']}"
        )

    print(f"训练 {len(train_indices)} 轮，留出 {len(test_indices)} 轮")
    print(line("默认 train", train_default))
    print(line("校准 train", train_calibrated))
    print(line("默认 test", test_default))
    print(line("校准 test", test_calibrated))
    print(f"\n阈值 -> {out_path}")
    print(f"报告 -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
