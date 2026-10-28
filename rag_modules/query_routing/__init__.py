"""
可迁移的查询路由包。

分层：
    domain.py        业务配置（意图、路由键、词表、图谱/向量库 Schema）
    prompts.py       业务 Prompt + 回答约束
    understanding.py 理解层编排（LLM 结构化输出 + 实体/上下文）
    rules.py         规则分类/追问/意图合并
    slots.py         槽位归一化/抽取
    policy.py        策略层（确定性选计划）
    evidence.py      执行层（Neo4j / 原文 / tips / hybrid）
    validators.py    覆盖校验
    tips_knowledge.py CookWise tips 知识库
    schemas.py       通用数据结构
    text_utils.py    纯文本工具
"""

from .domain import DOMAIN, DomainConfig, RouteKeySpec, RouteThreshold
from .evidence import EvidenceService, GraphEvidenceAdapter, TextEvidenceAdapter
from .policy import RoutePolicy
from .schemas import (
    ContextInfo,
    EntityRef,
    IntentItem,
    RoutePlan,
    SubPlan,
    Understanding,
    ValidationResult,
)
from .telemetry import RouteDecisionLogger
from .tips_knowledge import MealPlan, PairingEntry, SafetyEntry, TipsKnowledgeBase
from .understanding import QueryUnderstandingService
from .validators import EvidenceValidator

__all__ = [
    "DOMAIN",
    "DomainConfig",
    "RouteKeySpec",
    "RouteThreshold",
    "Understanding",
    "IntentItem",
    "EntityRef",
    "ContextInfo",
    "SubPlan",
    "RoutePlan",
    "ValidationResult",
    "QueryUnderstandingService",
    "RoutePolicy",
    "GraphEvidenceAdapter",
    "TextEvidenceAdapter",
    "EvidenceService",
    "EvidenceValidator",
    "RouteDecisionLogger",
    "TipsKnowledgeBase",
    "PairingEntry",
    "SafetyEntry",
    "MealPlan",
]
