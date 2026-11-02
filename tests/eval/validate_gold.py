#!/usr/bin/env python3
"""
离线校验黄金集：不连 Neo4j/Milvus，CI 和本地都能跑。

检查项：
- ID 唯一、分类标记一致、比例符合约定
- 12 个路由键各出现 30~100 次
- 正例实体能在 entities.json 里找到（菜名/食材/相克/食品安全/技法）
- 证据关键词与实体、路由键自洽
- response_type / needs_retrieval / 路由键互相不矛盾

用法：
    python tests/eval/validate_gold.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_modules.query_routing.domain import NO_RETRIEVAL_KEYS, ROUTE_SPECS

ALL_KEYS = list(ROUTE_SPECS)
IN_DOMAIN_KEYS = [key for key in ALL_KEYS if key not in NO_RETRIEVAL_KEYS]


class Checker:
    def __init__(self, entities: Dict[str, Any]):
        self.entities = entities
        self.recipes = {
            item["name"]: item for item in entities.get("recipes", [])
        }
        self.pairings = {
            (item["left"], item["right"]) for item in entities.get("pairings", [])
        }
        self.safety_sections = set(entities.get("safety_sections", []))
        self.techniques = set(entities.get("techniques", []))
        self.categories = set(entities.get("categories", []))
        self.cuisines = set(entities.get("cuisines", []))
        self.tags = set(entities.get("tags", []))
        self.errors: List[str] = []

    def error(self, case_id: str, message: str) -> None:
        self.errors.append(f"{case_id}: {message}")

    def check_case(self, case: Dict[str, Any]) -> None:
        case_id = str(case.get("id") or "")
        if not case_id:
            self.error("<no-id>", "缺少 id")
            return

        turns = case.get("turns") or []
        if not turns:
            self.error(case_id, "没有 turns")
            return

        category = case.get("category")
        multi_intent = bool(case.get("multi_intent"))
        multi_turn = bool(case.get("multi_turn"))
        expected_category = {
            (False, False): "single_turn_single_intent",
            (True, False): "single_turn_multi_intent",
            (False, True): "multi_turn_single_intent",
            (True, True): "multi_turn_multi_intent",
        }[(multi_intent, multi_turn)]
        if category != expected_category:
            self.error(case_id, f"category={category} 与标记不一致，应为 {expected_category}")
        if multi_turn != (len(turns) > 1):
            self.error(case_id, "multi_turn 标记与 turns 数量不一致")
        if multi_intent and not any(
            len(turn.get("expected", {}).get("route_keys", [])) > 1
            for turn in turns
        ):
            self.error(case_id, "multi_intent 标记为真，但没有多意图轮次")

        for index, turn in enumerate(turns):
            self.check_turn(case_id, index, turn)

    def check_turn(self, case_id: str, index: int, turn: Dict[str, Any]) -> None:
        query = str(turn.get("query") or "").strip()
        expected = turn.get("expected") or {}
        route_keys = list(expected.get("route_keys") or [])
        needs_retrieval = expected.get("needs_retrieval")
        response_type = expected.get("response_type")
        evidence = list(expected.get("evidence") or [])
        slots = expected.get("slots") or {}

        label = f"turn[{index}]"
        if not query:
            self.error(case_id, f"{label} query 为空")
        for key in route_keys:
            if key not in ROUTE_SPECS:
                self.error(case_id, f"{label} 未知路由键 {key}")

        if not isinstance(needs_retrieval, bool):
            self.error(case_id, f"{label} needs_retrieval 必须是布尔值")
        if response_type not in ("answer", "refuse", "clarify", "no_data"):
            self.error(case_id, f"{label} response_type 非法: {response_type}")

        retrieval_keys = [key for key in route_keys if key in IN_DOMAIN_KEYS]
        if route_keys and not retrieval_keys and needs_retrieval:
            self.error(case_id, f"{label} 只有非检索意图却标了需要检索")
        if retrieval_keys and not needs_retrieval:
            self.error(case_id, f"{label} 有检索意图却标了不需要检索")

        if response_type == "clarify":
            if route_keys or needs_retrieval:
                self.error(case_id, f"{label} 澄清轮不应有路由键/检索")
            if evidence:
                self.error(case_id, f"{label} 澄清轮不应有证据")
        if response_type == "refuse" and route_keys != ["out_of_scope"]:
            self.error(case_id, f"{label} 拒识轮应只有 out_of_scope")
        if response_type == "no_data" and route_keys != ["nutrition_value"]:
            self.error(case_id, f"{label} no_data 只用于 nutrition_value")
        if response_type == "answer" and not route_keys:
            self.error(case_id, f"{label} answer 轮缺少路由键")

        for item in evidence:
            self.check_evidence(case_id, label, item, route_keys)
        self.check_slots(case_id, label, slots, route_keys)

    def check_slots(self, case_id: str, label: str, slots: Dict[str, Any], route_keys: List[str]) -> None:
        if not slots:
            return
        if "category" in slots and str(slots["category"]) not in self.categories:
            self.error(case_id, f"{label} 未知菜类 {slots['category']}")
        if "cuisine" in slots and str(slots["cuisine"]) not in self.cuisines:
            self.error(case_id, f"{label} 未知菜系 {slots['cuisine']}")
        if "servings" in slots and not isinstance(slots["servings"], int):
            self.error(case_id, f"{label} servings 必须是整数")
        if "spicy" in slots and str(slots["spicy"]) not in ("不辣", "微辣", "中辣", "特辣"):
            self.error(case_id, f"{label} spicy 值非法 {slots['spicy']}")
        if slots and not any(key in ("recommend_list", "recommend_scenario") for key in route_keys):
            self.error(case_id, f"{label} 只有推荐类才应声明 slots")

    def check_evidence(
        self,
        case_id: str,
        label: str,
        item: Dict[str, Any],
        route_keys: List[str],
    ) -> None:
        route_key = str(item.get("route_key") or "")
        keywords = [str(word) for word in item.get("keywords") or []]
        recipe_names = [str(name) for name in item.get("recipe_names") or []]
        match = item.get("match", "all")
        forbidden = [str(word) for word in item.get("forbidden") or []]

        if route_key not in ROUTE_SPECS:
            self.error(case_id, f"{label} 证据路由键非法 {route_key}")
            return
        if route_key not in route_keys:
            self.error(case_id, f"{label} 证据路由键 {route_key} 不在期望路由键里")
        if match not in ("any", "all"):
            self.error(case_id, f"{label} evidence.match 非法: {match}")
        # 推荐类可以只用“存在候选菜谱文档”做证据，其余检索类必须有具体关键词
        if (
            route_key not in NO_RETRIEVAL_KEYS
            and route_key != "recommend_list"
            and not keywords
        ):
            self.error(case_id, f"{label} {route_key} 证据缺少 keywords")
        if forbidden and route_key != "recommend_list":
            self.error(case_id, f"{label} forbidden 只用于 recommend_list")

        for name in recipe_names:
            if name not in self.recipes:
                self.error(case_id, f"{label} 菜名不存在: {name}")

        if route_key == "recipe_ingredients":
            if not recipe_names:
                self.error(case_id, f"{label} 食材证据缺少菜名")
            for name in recipe_names:
                recipe = self.recipes.get(name, {})
                if keywords and keywords[0] not in recipe.get("ingredients", []):
                    self.error(case_id, f"{label} {name} 的食材不存在: {keywords[0]}")
            return

        if route_key == "recipe_steps_time":
            for name in recipe_names:
                recipe = self.recipes.get(name, {})
                if keywords and keywords[0] not in str(recipe.get("cook") or ""):
                    self.error(case_id, f"{label} {name} 的烹饪时间不匹配: {keywords[0]}")
            return

        if route_key == "recipe_overview":
            for name in recipe_names:
                recipe = self.recipes.get(name, {})
                if keywords and keywords[0] not in (
                    str(recipe.get("category") or "") + str(recipe.get("cuisine") or "")
                ):
                    self.error(case_id, f"{label} {name} 的分类/菜系不匹配: {keywords[0]}")
            return

        if route_key == "ingredient_pairing":
            if len(keywords) < 2 or (keywords[0], keywords[1]) not in self.pairings:
                self.error(case_id, f"{label} 相克条目不存在: {keywords}")
            return

        if route_key == "ingredient_property":
            if not any(word in self.techniques for word in keywords):
                self.error(case_id, f"{label} 技法不存在: {keywords}")
            return

        if route_key == "food_safety":
            if "食品安全" not in keywords:
                self.error(case_id, f"{label} 食品安全证据缺少“食品安全”")
            section = next((word for word in keywords if word != "食品安全"), "")
            if section not in self.safety_sections and not any(
                section in item or item in section for item in self.safety_sections
            ):
                self.error(case_id, f"{label} 食品安全段落不存在: {keywords}")
            return

        if route_key == "nutrition_value":
            for name in recipe_names:
                recipe = self.recipes.get(name, {})
                kcal = [str(value) for value in recipe.get("kcal", [])]
                if not any(word in kcal for word in keywords):
                    self.error(case_id, f"{label} {name} 原文热量不匹配: {keywords}")
            return

        if route_key == "nutrition_benefit":
            for name in recipe_names:
                recipe = self.recipes.get(name, {})
                term = str(recipe.get("benefit_term") or "")
                if term and not any(word in term for word in keywords):
                    self.error(case_id, f"{label} {name} 营养词不匹配: {keywords}")
            return

        if route_key == "recommend_list":
            normal_words = {"候选菜谱"}
            allowed_words = self.categories | self.cuisines | self.tags | normal_words
            # 空 keywords 表示只要求返回任意候选菜谱文档
            if keywords and not any(word in allowed_words for word in keywords):
                self.error(case_id, f"{label} 推荐证据词无法对应菜类/菜系/标签: {keywords}")
            return

        if route_key == "recommend_scenario":
            if "配菜规则" not in keywords:
                self.error(case_id, f"{label} 配菜规则证据缺少“配菜规则”")
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(ROOT / "tests/gold/golden_set.jsonl"))
    parser.add_argument("--entities", default=str(ROOT / "tests/gold/entities.json"))
    parser.add_argument("--min-cases", type=int, default=500)
    args = parser.parse_args()

    gold_path = Path(args.gold)
    entities_path = Path(args.entities)
    if not gold_path.exists():
        print(f"黄金集不存在: {gold_path}", file=sys.stderr)
        return 1
    if not entities_path.exists():
        print(f"实体快照不存在: {entities_path}", file=sys.stderr)
        return 1

    entities = json.loads(entities_path.read_text(encoding="utf-8"))
    checker = Checker(entities)

    cases: List[Dict[str, Any]] = []
    ids = set()
    with gold_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"第 {line_number} 行 JSON 解析失败: {exc}", file=sys.stderr)
                return 1
            case_id = str(case.get("id") or "")
            if not case_id:
                checker.error(f"<line-{line_number}>", "缺少 id")
            elif case_id in ids:
                checker.error(case_id, "id 重复")
            ids.add(case_id)
            cases.append(case)

    if len(cases) < args.min_cases:
        checker.error("<all>", f"只有 {len(cases)} 条，少于 {args.min_cases} 条")

    for case in cases:
        checker.check_case(case)

    key_counts: Counter = Counter()
    category_counts: Counter = Counter()
    for case in cases:
        category_counts[case.get("category", "unknown")] += 1
        for turn in case.get("turns") or []:
            key_counts.update((turn.get("expected") or {}).get("route_keys") or [])

    for key, count in key_counts.items():
        if key not in NO_RETRIEVAL_KEYS and not 30 <= count <= 100:
            checker.error("<coverage>", f"{key} 出现 {count} 次，不在 30~100")
    for key in ALL_KEYS:
        if key_counts[key] < 30:
            checker.error("<coverage>", f"{key} 只出现 {key_counts[key]} 次，少于 30")

    total = len(cases) or 1
    ratios = {
        "多意图": sum(1 for item in cases if item.get("multi_intent")) / total,
        "多轮": sum(1 for item in cases if item.get("multi_turn")) / total,
        "边界": sum(1 for item in cases if item.get("boundary")) / total,
    }
    target = {"多意图": 0.30, "多轮": 0.30, "边界": 0.20}
    for label, value in ratios.items():
        if abs(value - target[label]) > 0.02:
            checker.error("<ratio>", f"{label} 比例 {value:.1%} 偏离 {target[label]:.0%}")

    print(f"黄金集：{len(cases)} 条 -> {gold_path}")
    print("分类：", dict(category_counts))
    print("比例：", {label: f"{value:.1%}" for label, value in ratios.items()})
    print("路由键覆盖：")
    for key in ALL_KEYS:
        print(f"  {key:20s} {key_counts[key]}")

    if checker.errors:
        print(f"\n校验失败：{len(checker.errors)} 个问题", file=sys.stderr)
        for message in checker.errors[:80]:
            print(f"  - {message}", file=sys.stderr)
        if len(checker.errors) > 80:
            print(f"  ... 还有 {len(checker.errors) - 80} 个", file=sys.stderr)
        return 1

    print("\n校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
