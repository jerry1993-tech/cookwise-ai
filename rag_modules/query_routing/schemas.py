"""
查询路由的通用数据结构。

- LLM 返回的 JSON 先经过 ``understanding.py`` 做一次显式规范化；
- 路由层只依赖这些普通对象，换领域时不需要改数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EntityRef:
    """查询里提到的一个实体。"""

    name: str
    type: str = "unknown"          # recipe / ingredient / cuisine / category
    resolved_id: str = ""
    unresolved: bool = False


@dataclass
class IntentItem:
    """一个可独立路由的意图。多意图时就是列表里的多项。"""

    intent: str                    # 6 个顶层意图之一
    route_key: str                 # 12 个路由键之一
    evidence_profile: List[str] = field(default_factory=list)
    confidence: float = 0.5
    span: str = ""                 # 该意图对应的原文片段
    entities: List[EntityRef] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    need_complete: bool = False
    value_policy: str = ""         # exact_only / no_estimate
    expected_source: List[str] = field(default_factory=list)
    engine_override: str = ""      # 兼容旧版 LLM 输出时使用

    def entity_names(self, entity_type: str = "") -> List[str]:
        """取出实体名；可按类型过滤。"""
        return [
            item.name
            for item in self.entities
            if item.name and (not entity_type or item.type == entity_type)
        ]


@dataclass
class ContextInfo:
    """多轮上下文里与检索有关的状态。"""

    used_history: bool = False
    topic_shift: bool = False
    active_recipe: str = ""
    active_topic: str = ""
    resolved_entities: List[str] = field(default_factory=list)


@dataclass
class Understanding:
    """一次查询理解的完整结果，也是理解层与策略层之间的契约。"""

    standalone_query: str
    needs_retrieval: bool
    intents: List[IntentItem]
    context: ContextInfo = field(default_factory=ContextInfo)
    is_ambiguous: bool = False
    clarify_question: str = ""
    disclaimer_required: bool = False
    source: str = "llm"            # llm / rule / legacy
    raw: Dict[str, Any] = field(default_factory=dict)

    def top_intent(self, fallback_intent: str, fallback_route_key: str) -> IntentItem:
        """返回置信度最高的意图；空列表时用调用方传入的领域默认值。"""
        if not self.intents:
            return IntentItem(intent=fallback_intent, route_key=fallback_route_key)
        return max(self.intents, key=lambda item: item.confidence)

@dataclass
class SubPlan:
    """一个子问题的执行计划。"""

    intent: IntentItem
    query: str
    engine: str                    # graph_entity / graph_relation / hybrid / tips_* / combined / none
    top_k: int
    need_complete: bool
    validator: str
    depends_on: int = -1           # 依赖的子问题下标；-1 表示独立
    index: int = 0


@dataclass
class RoutePlan:
    """策略层的输出。"""

    needs_retrieval: bool
    sub_plans: List[SubPlan]
    response_type: str = "answer"  # answer / clarify / refuse
    clarify_question: str = ""
    is_multi_intent: bool = False


@dataclass
class ValidationResult:
    """证据覆盖校验结果。"""

    status: str                    # complete / partial / missing
    detail: str = ""
    retry_hint: str = ""
