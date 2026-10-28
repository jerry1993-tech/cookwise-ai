"""
不依赖 LLM 的规则分类。

输入是用户原句和会话上下文，输出是 IntentItem；所有业务词表来自 domain.py。
LLM 解析失败、JSON 不合法、或者没有 LLM 时，都走这里。
"""

from typing import Any, Callable, Dict, List, Optional

from .domain import DomainConfig
from .schemas import EntityRef, IntentItem
from .slots import SlotExtractor
from .text_utils import clamp_float, has_reference, normalize_text


class RuleClassifier:
    """按领域配置做追问分类、规则路由和意图合并。"""

    def __init__(
        self,
        domain: DomainConfig,
        entity_extractor: Callable[[str], List[EntityRef]],
        slot_extractor: SlotExtractor,
    ):
        self.domain = domain
        self.extract_entities = entity_extractor
        self.slot_extractor = slot_extractor

    def classify_follow_up(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> Optional[IntentItem]:
        """
        有 active_recipe 且出现指代/省略时，把追问归到“那道菜”的某个属性上。

        典型：先问宫保鸡丁，再问“它需要哪些食材”“那要多少盐”“要煮多久”。
        """
        active_recipe = str(context.get("active_recipe") or "")
        if not active_recipe or active_recipe in text:
            return None
        if not has_reference(text, self.domain.reference_terms):
            return None

        for rule in self.domain.follow_up_rules:
            if rule.terms and not any(term in text for term in rule.terms):
                continue
            slots = dict(rule.slots)
            for slot_name, terms in rule.slot_from_terms.items():
                for term in terms:
                    if term in text:
                        slots[slot_name] = term
                        break
            return self.make_intent(rule.route_key, text, slots=slots)
        return None

    def apply_wording_rules(self, route_key: str, text: str) -> str:
        """按领域配置改写措辞带来的路由变化（如“做法”归概览）。"""
        for rule in self.domain.wording_rules:
            if route_key not in rule.source_keys:
                continue
            if not any(term in text for term in rule.terms):
                continue
            if any(term in text for term in rule.exclude_terms):
                continue
            return rule.target_key
        return route_key

    def rule_classify(self, text: str) -> Optional[IntentItem]:
        """按 domain.rule_specs 的顺序匹配；返回 None 表示这段没有明确意图。"""
        text = normalize_text(text)
        if not text:
            return None

        request_hit = any(term in text for term in self.domain.request_terms)
        for rule in self.domain.rule_specs:
            if rule.require_request and not request_hit:
                continue
            if rule.skip_if_request and request_hit:
                continue
            if any(term in text for term in rule.skip_if_any):
                continue
            if rule.terms and not any(term in text for term in rule.terms):
                continue
            return self.make_intent(rule.route_key, text, slots=dict(rule.slots))
        return None

    def default_intent(self, query: str) -> IntentItem:
        """没有明确意图时：有饮食词就走默认路由，否则判域外。"""
        if any(term in query for term in self.domain.food_hints):
            return self.make_intent(
                self.domain.default_route_key, query,
                confidence=0.55, engine_override="hybrid",
            )
        return self.make_intent(
            self.domain.out_of_scope_key, query, confidence=0.6
        )

    def make_intent(
        self,
        route_key: str,
        text: str,
        confidence: float = 0.75,
        slots: Optional[Dict[str, Any]] = None,
        engine_override: str = "",
    ) -> IntentItem:
        """按路由键定义生成一个规范化意图。"""
        spec = self.domain.spec(route_key)
        entities = self.extract_entities(text)
        merged_slots = self.slot_extractor.extract(text)
        if slots:
            merged_slots.update(slots)

        # 有实体且路由配置要求时，置信度略高
        if entities and spec.boost_with_entity:
            confidence = max(confidence, 0.8)

        return IntentItem(
            intent=spec.intent,
            route_key=route_key,
            evidence_profile=list(spec.evidence_profile),
            confidence=clamp_float(confidence),
            span=text,
            entities=entities,
            slots=merged_slots,
            need_complete=spec.need_complete,
            value_policy=spec.value_policy,
            expected_source=[spec.source],
            engine_override=engine_override,
        )

    def merge_intents(self, items: List[IntentItem]) -> List[IntentItem]:
        """
        同路由键的追问合并，但“两道不同的菜分别问同一属性”要保留成两个意图。

        例：
        - “8人聚餐，荤素怎么配？要加鱼吗” -> 合并成一个 scenario
        - “宫保鸡丁要哪些食材？可乐鸡翅要哪些食材” -> 保留两个 ingredients
        """
        merged: List[IntentItem] = []

        for item in items:
            target = None
            for existing in merged:
                if existing.route_key != item.route_key:
                    continue
                existing_recipes = set(existing.entity_names("recipe"))
                item_recipes = set(item.entity_names("recipe"))
                if existing_recipes and item_recipes and not (existing_recipes & item_recipes):
                    continue  # 同一路由键，但不是同一道菜
                target = existing
                break

            if target is None:
                merged.append(item)
                continue

            if item.span and item.span not in target.span:
                target.span = f"{target.span}；{item.span}"
            target.confidence = max(target.confidence, item.confidence)
            target.need_complete = target.need_complete or item.need_complete
            target.slots.update(item.slots)

            known_names = {entity.name for entity in target.entities}
            for entity in item.entities:
                if entity.name not in known_names:
                    target.entities.append(entity)
                    known_names.add(entity.name)

        return merged

    def infer_route_key(self, text: str) -> str:
        """LLM 给了非法 route_key 时，用规则分类兜底。"""
        item = self.rule_classify(text)
        return item.route_key if item else self.domain.default_route_key

    def pure_constraint(self, query: str) -> Optional[IntentItem]:
        """纯约束声明（如“我不吃辣”）不检索，只记录到会话约束。"""
        spec = self.domain.constraint_spec
        if spec is None:
            return None
        if any(term in query for term in spec.terms) and not any(
            term in query for term in spec.request_terms
        ):
            return self.make_intent(
                spec.route_key, query, confidence=spec.confidence,
                slots=dict(spec.slots),
            )
        return None
