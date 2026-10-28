"""
槽位归一化与抽取。

把 LLM 返回的字段名和口语值收敛成内部槽位，供策略层和候选过滤使用。
业务词表全部来自 domain.SlotVocabulary；本文件只负责匹配和清洗。
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from .domain import DomainConfig


class SlotExtractor:
    """按领域词表归一化/抽取槽位。"""

    def __init__(self, domain: DomainConfig, cuisines: Sequence[str] = ()):
        self.domain = domain
        self.cuisine_set = set(cuisines or ())

    def set_cuisines(self, cuisines: Sequence[str]) -> None:
        """实体词表刷新后同步菜系集合。"""
        self.cuisine_set = set(cuisines or ())

    def normalize(self, slots: Any, route_key: str = "") -> Dict[str, Any]:
        """把 LLM 常用字段名映射成内部槽位键，并做类型收敛。"""
        if not isinstance(slots, dict):
            return {}

        result = dict(slots)
        vocab = self.domain.slot_vocabulary
        aliases = {
            "people_count": "servings",
            "peopleCount": "servings",
            "number_of_people": "servings",
            "headcount": "servings",
            "people": "servings",
            "goal": "diet",
            "diet_goal": "diet",
            "dietGoal": "diet",
            "spicy_level": "spicy",
            "spicyLevel": "spicy",
            "cook_time": "max_minutes",
            "max_time": "max_minutes",
            "maxMinutes": "max_minutes",
            "time_limit": "max_minutes",
            "with_child": "has_child",
            "hasChild": "has_child",
            "is_child": "has_child",
            "cuisine_type": "cuisine",
            "cuisineType": "cuisine",
        }
        for source_key, target_key in aliases.items():
            if source_key in result and target_key not in result:
                result[target_key] = result[source_key]

        for key in ("servings", "max_minutes"):
            if key in result:
                match = re.search(r"\d+", str(result[key]))
                if match:
                    result[key] = int(match.group())

        if "has_child" in result:
            value = result["has_child"]
            if isinstance(value, str):
                result["has_child"] = value.strip().lower() in (
                    "true", "1", "yes", "是", "有", "要",
                )
            else:
                result["has_child"] = bool(value)

        # 只有推荐/筛选类意图才把口味当过滤条件；问菜谱详情时它只是菜名描述
        spec = self.domain.spec(route_key) if route_key else None
        if spec is None or "FILTER" in spec.evidence_profile:
            for key in ("spicy", "taste", "flavor"):
                canonical = self.canonical_from(result.get(key), vocab.spicy_aliases)
                if canonical:
                    result["spicy"] = canonical
                    break
            for key in ("taste", "flavor"):
                if self.canonical_from(result.get(key), vocab.spicy_aliases):
                    result.pop(key, None)
                else:
                    # 甜/酸/咸等口味也收敛成图谱里的标准词
                    taste = self.canonical_from(result.get(key), vocab.taste_terms)
                    if taste:
                        result[key] = taste

        # exclude 统一成列表，兼容 LLM 的字符串/列表两种写法
        exclude = result.get("exclude")
        if isinstance(exclude, str):
            exclude = re.split(r"[、,，;；\s]+", exclude)
        if isinstance(exclude, (list, tuple)):
            cleaned = [self.clean_exclude_term(part) for part in exclude]
            result["exclude"] = [part for part in cleaned if part]

        # diet 里的“素菜/荤菜”是菜类，不是减脂目标，移到 category
        diet = str(result.get("diet") or "").strip()
        if diet:
            category = self.domain.category_aliases.get(diet)
            if category:
                result.setdefault("category", category)
                if not any(term in diet for term in self.domain.diet_terms):
                    result.pop("diet", None)
            elif not any(term in diet for term in self.domain.diet_terms):
                # 认不出的 diet 不做过滤，避免把“清淡”误当成减脂
                result.pop("diet", None)

        # category 统一到图谱真实分类；写成菜系的挪到 cuisine，写成“快手”的挪到 time
        category = str(result.get("category") or "").strip()
        if category:
            # “家常菜”这类口语词不在图谱 cuisineType 里，不能拿它做过滤
            known_cuisines = self.cuisine_set or set(self.domain.cuisine_terms)
            if category in self.domain.category_aliases:
                result["category"] = self.domain.category_aliases[category]
            elif category in known_cuisines:
                result.setdefault("cuisine", category)
                result.pop("category", None)
            elif category in vocab.quick_terms:
                result.setdefault("time", vocab.quick_value)
                result.pop("category", None)
            else:
                # 不在词表里的分类不做过滤，交给文本兜底
                result.pop("category", None)

        return result

    @staticmethod
    def plain_taste(value: Any) -> str:
        """去掉“的/口味/味道”等后缀，方便和口味词表对齐。"""
        text = str(value or "").strip().rstrip("的")
        for suffix in ("口味", "味道"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
        return text

    @staticmethod
    def clean_exclude_term(value: Any) -> str:
        """去掉排除项尾部的“的菜/的菜谱/的”等口语后缀。"""
        text = str(value or "").strip()
        for suffix in ("的菜谱", "的菜品", "的菜", "的"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break
        return text.strip().rstrip("的").strip()

    @staticmethod
    def hits_any(value: Any, terms: Sequence[str]) -> bool:
        """槽位值（可能是列表）是否命中任一排除词。"""
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            text = str(item).strip()
            if text and any(text == term or text in term or term in text for term in terms):
                return True
        return False

    @staticmethod
    def merge_exclude(*groups: Any) -> List[str]:
        """把多轮排除项合并去重；兼容字符串和列表。"""
        merged: List[str] = []
        for group in groups:
            if isinstance(group, str):
                group = [group]
            for item in group or []:
                text = str(item).strip()
                if text and text not in merged:
                    merged.append(text)
        return merged

    def canonical_from(
        self,
        value: Any,
        table: Sequence[Tuple[str, Tuple[str, ...]]],
    ) -> str:
        """在 (规范值, 同义说法) 表里找完全匹配，避免“辣椒”误判成辣度。"""
        text = self.plain_taste(value)
        if not text:
            return ""
        for canonical, terms in table:
            if text == canonical or text in terms:
                return canonical
        return ""

    def extract(self, text: str) -> Dict[str, Any]:
        """从文本里抽约束槽位；抽不到就不写，保持空对象。"""
        slots: Dict[str, Any] = {}
        vocab = self.domain.slot_vocabulary

        # diet / persona 都是“先命中先写”，后面的同槽位词不再覆盖
        for value, terms in vocab.diet:
            if any(term in text for term in terms):
                slots["diet"] = value
                break

        for value, terms in vocab.persona:
            if any(term in text for term in terms):
                slots["persona"] = value
                break

        if any(term in text for term in vocab.quick_terms):
            slots["time"] = vocab.quick_value

        minute_match = re.search(vocab.minutes_pattern, text)
        if minute_match:
            slots["max_minutes"] = int(minute_match.group(1))

        servings_match = re.search(vocab.servings_pattern, text)
        if servings_match:
            slots["servings"] = int(servings_match.group(1))

        for spicy, terms in vocab.spicy_text_terms:
            if any(term in text for term in terms):
                slots["spicy"] = spicy
                break
        if any(term in text for term in vocab.constraint_terms):
            slots["spicy"] = vocab.spicy_fallback

        for cuisine in self.domain.cuisine_terms:
            if cuisine in text:
                slots["cuisine"] = cuisine
                break

        # “推荐几道素菜/汤类”这类按分类筛选，候选查询要用 category
        for category in vocab.category_terms:
            if category in text:
                slots["category"] = category
                break

        # 甜/酸/咸等非辣口味，交给候选查询的 flavor 过滤
        for taste, terms in vocab.taste_terms:
            if any(term in text for term in terms):
                slots["flavor"] = taste
                break

        if vocab.exclude_pattern:
            exclude_match = re.search(vocab.exclude_pattern, text)
            if exclude_match:
                exclude = self.clean_exclude_term(exclude_match.group(1))
                if exclude:
                    slots["exclude"] = [exclude]

        # “带辣椒的菜 / 含牛肉的菜”抽成食材；如果是“麻辣/甜味”则归到口味
        if vocab.ingredient_pattern:
            ingredient_match = re.search(vocab.ingredient_pattern, text)
            if ingredient_match:
                term = re.sub(r"^[点些]", "", ingredient_match.group(1)).strip()
                spicy = self.canonical_from(term, vocab.spicy_aliases)
                taste = self.canonical_from(term, vocab.taste_terms)
                if spicy:
                    slots.setdefault("spicy", spicy)
                elif taste:
                    slots.setdefault("flavor", taste)
                else:
                    parts = [part for part in re.split(r"[和与及、]", term) if part]
                    if parts:
                        slots["ingredient"] = parts[0] if len(parts) == 1 else parts

        # 否定偏好与正向槽位互斥：“不要甜品”不能再按甜品筛选
        excludes = slots.get("exclude") or []
        if excludes:
            for key in (
                "diet", "persona", "category", "flavor", "taste",
                "ingredient", "cuisine", "spicy",
            ):
                if key in slots and self.hits_any(slots[key], excludes):
                    slots.pop(key)
            if any(
                self.canonical_from(term, vocab.spicy_aliases) for term in excludes
            ):
                slots["spicy"] = vocab.spicy_fallback

        if any(term in text for term in vocab.add_fish_terms) or (
            vocab.add_fish_all_terms
            and all(term in text for term in vocab.add_fish_all_terms)
        ):
            slots["add_fish"] = True
        if any(term in text for term in vocab.child_terms):
            slots["has_child"] = True

        return slots
