"""
策略层：把 Understanding 变成确定性的 RoutePlan。

LLM 不参与这里。这样路由可以单测、可以解释，也不会出现
“同一个问题两次调用给出两个不同引擎”的情况。
"""

import logging
from dataclasses import replace
from typing import List, Optional

from .domain import DOMAIN, DomainConfig, RouteThreshold
from .schemas import EntityRef, IntentItem, RoutePlan, SubPlan, Understanding

logger = logging.getLogger(__name__)


class RoutePolicy:
    """确定性路由策略。"""

    def __init__(self, domain: DomainConfig = DOMAIN):
        self.domain = domain

    def _threshold(self, route_key: str) -> RouteThreshold:
        """取路由键阈值。"""
        return self.domain.threshold(route_key)

    def plan(self, understanding: Understanding, top_k: int = 5) -> RoutePlan:
        """生成执行计划。"""
        intents = [
            item for item in understanding.intents
            if item.route_key in self.domain.route_specs
        ]
        if not intents:
            return self._clarify("no_intent")

        if understanding.is_ambiguous:
            return self._clarify("ambiguous", understanding.clarify_question)

        # 一个意图里点了两道菜（“宫保鸡丁和可乐鸡翅要哪些食材”）时拆成两个子计划
        intents = self._expand_multi_recipe(intents)

        retrieval_intents = [
            item for item in intents
            if item.route_key not in self.domain.no_retrieval_keys
        ]
        no_retrieval_intents = [
            item for item in intents
            if item.route_key in self.domain.no_retrieval_keys
        ]

        needs_retrieval = self._decide_retrieval(
            understanding, retrieval_intents, no_retrieval_intents
        )
        if not needs_retrieval:
            return self._plan_direct(intents, no_retrieval_intents, understanding)

        top_retrieval = max(retrieval_intents, key=lambda item: item.confidence)
        if top_retrieval.confidence < self._threshold(top_retrieval.route_key).min_execute:
            return self._clarify("low_confidence")

        return self._plan_retrieval(intents, understanding, top_k)

    def _clarify(self, kind: str, question: str = "") -> RoutePlan:
        """统一的澄清计划；话术由领域配置提供。"""
        return RoutePlan(
            needs_retrieval=False,
            sub_plans=[],
            response_type="clarify",
            clarify_question=question or self.domain.clarify_messages[kind],
        )

    def _plan_direct(
        self,
        intents: List[IntentItem],
        no_retrieval_intents: List[IntentItem],
        understanding: Understanding,
    ) -> RoutePlan:
        """不检索：选置信度最高的意图，直接回答或拒绝。"""
        top = max(no_retrieval_intents or intents, key=lambda item: item.confidence)
        sub_plan = self._build_sub_plan(0, top, understanding, "none", 0, is_multi=False)
        return RoutePlan(
            needs_retrieval=False,
            sub_plans=[sub_plan],
            response_type="refuse" if self.domain.spec(top.route_key).refuse else "answer",
        )

    def _plan_retrieval(
        self,
        intents: List[IntentItem],
        understanding: Understanding,
        top_k: int,
    ) -> RoutePlan:
        """需要检索：为每个意图构造子计划。"""
        # 多个意图项就是多意图；同一路由键但不同菜名（两道菜分别问）也算
        is_multi = len(intents) > 1
        shared_recipe = self._shared_recipe(intents)
        sub_plans: List[SubPlan] = []

        for index, item in enumerate(intents):
            spec = self.domain.spec(item.route_key)

            # 多意图里省略主语的子问题（“要煮多久”）补上共享菜名，证据层才能绑定
            if shared_recipe and spec.entity_first and not item.entity_names("recipe"):
                item = replace(
                    item,
                    entities=[EntityRef(name=shared_recipe, type="recipe"), *item.entities],
                )

            engine = item.engine_override or spec.engine
            if item.route_key in self.domain.no_retrieval_keys:
                engine = "none"

            query = self._sub_query(item, understanding, is_multi, shared_recipe)
            sub_plans.append(self._build_sub_plan(
                index=index,
                item=item,
                understanding=understanding,
                engine=engine,
                top_k=self._top_k(spec, engine, top_k),
                is_multi=is_multi,
                query=query,
                depends_on=self._depends_on(intents, index),
            ))

        return RoutePlan(
            needs_retrieval=True,
            sub_plans=sub_plans,
            response_type="answer",
            is_multi_intent=is_multi,
        )

    # ------------------------------------------------------------------
    # 决策辅助
    # ------------------------------------------------------------------

    def _expand_multi_recipe(self, intents: List[IntentItem]) -> List[IntentItem]:
        """一个“菜谱属性”意图里含多个菜名时，拆成每道菜一个子计划。"""
        expanded: List[IntentItem] = []
        for item in intents:
            spec = self.domain.spec(item.route_key)
            recipes = item.entity_names("recipe")
            if spec.expand_per_recipe and len(recipes) > 1:
                for name in recipes:
                    expanded.append(replace(
                        item,
                        entities=[EntityRef(name=name, type="recipe")],
                        slots=dict(item.slots),
                    ))
            else:
                expanded.append(item)
        return expanded

    def _decide_retrieval(
        self,
        understanding: Understanding,
        retrieval_intents: List[IntentItem],
        no_retrieval_intents: List[IntentItem],
    ) -> bool:
        """综合 LLM 判断与置信度，决定是否检索；阈值按路由键取。"""
        # 任一检索意图达到自己的阈值，就覆盖 LLM 的 false，避免静默丢召回
        if any(
            item.confidence >= self._threshold(item.route_key).retrieval_override
            for item in retrieval_intents
        ):
            return True

        if not retrieval_intents:
            return False

        # 闲聊/域外意图足够确定时，允许拦截检索
        if any(
            item.confidence >= self._threshold(item.route_key).no_retrieval
            for item in no_retrieval_intents
        ):
            return False

        return bool(understanding.needs_retrieval)

    def _build_sub_plan(
        self,
        index: int,
        item: IntentItem,
        understanding: Understanding,
        engine: str,
        top_k: int,
        is_multi: bool,
        query: Optional[str] = None,
        depends_on: int = -1,
    ) -> SubPlan:
        """构造单个子计划。"""
        return SubPlan(
            intent=item,
            query=query or understanding.standalone_query,
            engine=engine,
            top_k=max(0, top_k),
            need_complete=item.need_complete,
            validator=self.domain.spec(item.route_key).validator,
            depends_on=depends_on,
            index=index,
        )

    def _sub_query(
        self,
        item: IntentItem,
        understanding: Understanding,
        is_multi: bool,
        shared_recipe: str,
    ) -> str:
        """多意图用 span 作为子查询；span 缺实体时补上共享菜名。"""
        if not is_multi:
            return understanding.standalone_query

        query = item.span.strip() or understanding.standalone_query
        # 只在子问题自己没有菜名时补共享菜名；推荐类由配置关闭
        spec = self.domain.spec(item.route_key)
        if (
            shared_recipe
            and spec.prefix_shared_recipe
            and not item.entity_names("recipe")
            and shared_recipe not in query
        ):
            query = f"{shared_recipe} {query}"
        return query

    @staticmethod
    def _shared_recipe(intents: List[IntentItem]) -> str:
        """取所有意图里出现的第一个菜谱实体，用于补齐省略的子问题。"""
        for item in intents:
            for name in item.entity_names("recipe"):
                return name
        return ""

    def _top_k(self, spec, engine: str, top_k: int) -> int:
        """按引擎决定返回条数；清单类由适配器按图谱实际数量再放大。"""
        if engine == "none":
            return 0
        if engine.startswith("tips_"):
            return max(1, spec.default_top_k)
        if engine in ("graph_entity",):
            return max(spec.default_top_k, top_k)
        return max(1, top_k)

    def _depends_on(self, intents: List[IntentItem], index: int) -> int:
        """
        标记生成阶段的依赖关系。

        检索仍然并行执行（这些子问题各自都能独立检索），
        依赖只影响生成时“先答哪个、后答哪个”。
        """
        if index <= 0:
            return -1
        previous = intents[index - 1]
        current = intents[index]

        if previous.route_key in self.domain.spec(current.route_key).depends_on_previous:
            return index - 1
        return -1
