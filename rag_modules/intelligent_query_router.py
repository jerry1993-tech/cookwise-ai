"""
智能查询路由器。

对外仍然只有原来的入口：
    analyze_query(query) -> QueryAnalysis
    route_query(query, top_k) -> (documents, QueryAnalysis)

内部改成三层：
    理解层（LLM 结构化输出 + 规则兜底）
      -> 策略层（确定性 route_key -> plan）
      -> 执行层（图谱 / 原文 / tips / hybrid）

保留 ``recommended_strategy`` 等旧字段，避免 main.py / 测试 / 前端受影响。
"""

import concurrent.futures
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .query_routing import (
    DOMAIN,
    DomainConfig,
    EvidenceService,
    EvidenceValidator,
    GraphEvidenceAdapter,
    QueryUnderstandingService,
    RouteDecisionLogger,
    RoutePolicy,
    TextEvidenceAdapter,
    TipsKnowledgeBase,
)
from .query_routing.schemas import RoutePlan, SubPlan, Understanding, ValidationResult
from .query_routing.text_utils import normalize_text

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """搜索策略枚举；前三个保持旧含义，后三个用于新路由键。"""

    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"
    TIPS_KNOWLEDGE = "tips_knowledge"
    RULE_ENGINE = "rule_engine"
    NO_RETRIEVAL = "no_retrieval"


# engine -> route_stats 键；engine 为空时再按旧 strategy 兜底
ENGINE_STAT_KEYS = {
    "graph_entity": "graph_rag_count",
    "graph_relation": "graph_rag_count",
    "hybrid": "traditional_count",
    "combined": "combined_count",
    "tips_pairing": "tips_count",
    "tips_safety": "tips_count",
    "tips_technique": "tips_count",
    "tips_rule": "rule_count",
    "none": "no_retrieval_count",
}
STRATEGY_STAT_KEYS = {
    SearchStrategy.HYBRID_TRADITIONAL: "traditional_count",
    SearchStrategy.GRAPH_RAG: "graph_rag_count",
    SearchStrategy.COMBINED: "combined_count",
    SearchStrategy.TIPS_KNOWLEDGE: "tips_count",
    SearchStrategy.RULE_ENGINE: "rule_count",
    SearchStrategy.NO_RETRIEVAL: "no_retrieval_count",
}


@dataclass
class QueryAnalysis:
    """查询分析结果；旧字段保持不变，新字段都有默认值。"""

    # --- 旧接口字段（main.py / 前端 / 测试仍可能读取） ---
    query_complexity: float             # 查询复杂度 0~1；多意图、多实体会更高
    relationship_intensity: float       # 关系推理强度 0~1；REL 类为 1，其余 0.5
    reasoning_required: bool            # 是否需要多跳或关系推理
    entity_count: int                   # 去重后的实体数量
    recommended_strategy: SearchStrategy  # 由 engine 映射出的旧搜索策略
    confidence: float                   # 主意图置信度 0~1
    reasoning: str                      # 一句话路由解释

    # --- 理解层结果 ---
    # 指代消解/改写后的独立查询；检索与缓存都用它
    standalone_query: str = ""
    # 是否需要检索；闲聊、域外、澄清为 False
    needs_retrieval: bool = True
    # 主意图：6 个顶层意图之一
    intent: str = ""
    # 细路由键：12 个路由键之一
    route_key: str = ""
    # 主执行引擎：graph_entity / graph_relation / hybrid / tips_* / combined / none
    engine: str = ""
    # 期望证据形态：LIST / REL / TEXT / VALUE / FILTER / RULE / NONE
    evidence_profile: List[str] = field(default_factory=list)
    # 去重后的实体名
    entities: List[str] = field(default_factory=list)
    # 主意图槽位：菜系 / 分类 / 辣度 / 食材 / 排除项等
    slots: Dict[str, Any] = field(default_factory=dict)
    # 期望证据来源，如 graph:REQUIRES 全量 + 菜谱原文
    expected_source: List[str] = field(default_factory=list)
    # 完整理解结果（Understanding 的 asdict），含 intents / context / raw
    understanding: Optional[Dict[str, Any]] = None

    # --- 策略层计划 ---
    # 执行计划（RoutePlan）：子计划、引擎、top_k、依赖关系
    plan: Optional[RoutePlan] = None
    # 子问题摘要：index / intent / route_key / query / engine / entities 等
    sub_queries: List[Dict[str, Any]] = field(default_factory=list)
    # 是否拆分出多个独立意图
    is_multi_intent: bool = False
    # 回答类型：answer / clarify / refuse / no_data（no_data 在检索后置）
    response_type: str = "answer"
    # 需要澄清时反问用户的问题
    clarify_question: str = ""

    # --- 执行层结果 ---
    # 整体证据状态：complete / partial / missing / not_required / fallback
    evidence_status: str = ""
    # 每个子计划的证据明细：route_key / engine / status / detail / doc_count
    sub_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # --- 多轮上下文与生成标记 ---
    # 会话/缓存主题键：当前菜名、食材对或沿用中的菜谱
    topic_id: str = ""
    # 本轮是否用了多轮上下文（指代消解、约束继承）
    used_history: bool = False
    # 本轮是否切换主题，旧菜谱上下文不再沿用
    topic_shift: bool = False
    # 回答是否需要食品安全免责声明
    disclaimer_required: bool = False


class IntelligentQueryRouter:
    """智能查询路由器。"""

    def __init__(self, traditional_retrieval, graph_rag_retrieval, llm_client, config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        # 路由统计：保留旧键，新增 tips / 规则 / 不检索计数
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "tips_count": 0,
            "rule_count": 0,
            "no_retrieval_count": 0,
            "total_queries": 0,
        }

        # 决策落盘：本地 JSONL，供影子回放 / 阈值校准 / canary 监控离线读取
        self.decision_logger = RouteDecisionLogger(
            path=getattr(config, "route_log_path", None),
            enabled=getattr(config, "route_log_enabled", None),
        )

        # 业务适配器
        self.data_module = getattr(traditional_retrieval, "data_module", None)
        self.graph_adapter = GraphEvidenceAdapter(
            graph_retriever=graph_rag_retrieval,
            data_module=self.data_module,
        )
        self.tips_kb = TipsKnowledgeBase()
        self.text_adapter = TextEvidenceAdapter(traditional_retrieval, self.graph_adapter)
        self.evidence_service = EvidenceService(
            graph_adapter=self.graph_adapter,
            text_adapter=self.text_adapter,
            tips=self.tips_kb,
            graph_retriever=graph_rag_retrieval,
        )
        self.validator = EvidenceValidator()
        self.policy = RoutePolicy(DOMAIN)

        # 实体词表在图谱加载完成后才可用，首次查询时刷新一次
        self.understanding_service = QueryUnderstandingService(
            llm_client=llm_client,
            llm_model=getattr(config, "llm_model", "deepseek-flash"),
            domain=DOMAIN,
        )
        self._entities_loaded = False

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def analyze_query(
        self,
        query: str,
        session_id: str = "",
        session_state: Optional[Dict[str, Any]] = None,
    ) -> QueryAnalysis:
        """深度分析查询，输出理解结果 + 确定性执行计划。"""
        query = normalize_text(query)
        logger.info("分析查询: %s", query)

        self._ensure_entities()
        state = session_state or {}
        history_text = self._history_text(state)

        try:
            understanding = self.understanding_service.understand(
                query, history_text=history_text, context=state
            )
            plan = self.policy.plan(
                understanding,
                top_k=int(getattr(self.config, "top_k", 5) or 5),
            )
            analysis = self._build_analysis(query, understanding, plan)
            logger.info(
                "查询理解完成: intent=%s route_key=%s engine=%s needs_retrieval=%s "
                "confidence=%.2f source=%s",
                analysis.intent,
                analysis.route_key,
                analysis.engine,
                analysis.needs_retrieval,
                analysis.confidence,
                (analysis.understanding or {}).get("source", ""),
            )
            return analysis
        except Exception as exc:
            logger.error("查询分析失败，走规则兜底: %s", exc)
            return self._rule_based_analysis(query)

    def route_query(
        self,
        query: str,
        top_k: int = 5,
        session_id: str = "",
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Document], QueryAnalysis]:
        """智能路由查询；返回 (证据文档, 分析结果)。"""
        logger.info("开始智能路由: %s", query)
        started = time.perf_counter()
        analysis = self.analyze_query(query, session_id=session_id, session_state=session_state)
        self._update_route_stats(analysis.recommended_strategy, analysis.engine)

        # 澄清 / 域外拒识：不检索，交给生成层直接回答
        if analysis.response_type in ("clarify", "refuse") or not analysis.needs_retrieval:
            self._log_decision(
                query, analysis, [], "not_required", [], started, session_id
            )
            return [], analysis

        plan = analysis.plan
        if plan is None or not plan.sub_plans:
            self._log_decision(
                query, analysis, [], "missing", [], started, session_id,
                error="执行计划为空",
            )
            return [], analysis

        try:
            documents, status, sub_evidence = self._execute_plan(plan)

            # 发生过改写且证据仍不完整时，用原始 query 再检索一次做兜底
            if (
                analysis.used_history
                and analysis.standalone_query != normalize_text(query)
                and status != "complete"
            ):
                extra = self.text_adapter.search(query, top_k)
                if extra:
                    documents = self._merge_documents(documents, extra)
                    status = "partial"

            analysis.evidence_status = status
            analysis.sub_evidence = sub_evidence

            # 只要所有子问题的证据都缺失，就明确走“无数据”，
            # 不能因为兜底文本检索拿回几条无关文档就当作可回答
            all_missing = bool(sub_evidence) and all(
                item.get("status") == "missing" for item in sub_evidence
            )
            if analysis.response_type == "answer" and (not documents or all_missing):
                analysis.response_type = "no_data"
                analysis.reasoning = f"{analysis.reasoning}；未检索到证据"

            documents = self._post_process_results(documents, analysis)
            logger.info(
                "路由完成，返回 %d 个结果，证据状态=%s", len(documents), status
            )
            self._log_decision(
                query, analysis, documents, status, sub_evidence, started, session_id
            )
            return documents, analysis

        except Exception as exc:
            logger.error("查询路由失败，降级到传统检索: %s", exc)
            documents = self._fallback_search(query, top_k)
            analysis.engine = "hybrid"
            analysis.recommended_strategy = SearchStrategy.HYBRID_TRADITIONAL
            analysis.evidence_status = "fallback"
            analysis.sub_evidence = []
            analysis.reasoning = f"{analysis.reasoning}；路由异常，已降级"
            self._log_decision(
                query, analysis, documents, "fallback", [], started, session_id,
                error=str(exc),
            )
            return self._post_process_results(documents, analysis), analysis

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """规则兜底分析；保留方法名以兼容旧调用。"""
        understanding = self.understanding_service.rule_understand(query)
        plan = self.policy.plan(
            understanding,
            top_k=int(getattr(self.config, "top_k", 5) or 5),
        )
        return self._build_analysis(query, understanding, plan)

    # ------------------------------------------------------------------
    # 计划执行
    # ------------------------------------------------------------------

    def _execute_plan(
        self, plan: RoutePlan
    ) -> Tuple[List[Document], str, List[Dict[str, Any]]]:
        """并行执行子计划，合并去重，返回整体状态和每个子计划的证据状态。"""
        sub_plans = plan.sub_plans
        if len(sub_plans) == 1:
            results = [self._execute_sub_plan(sub_plans[0])]
        else:
            results: List[Tuple[List[Document], ValidationResult]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(sub_plans))) as executor:
                futures = [executor.submit(self._execute_sub_plan, item) for item in sub_plans]
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        logger.warning("子计划执行失败: %s", exc)
                        results.append(([], ValidationResult("missing", "子计划执行失败")))

        merged: List[Document] = []
        seen = set()
        statuses: List[str] = []
        details: List[Dict[str, Any]] = []

        for sub_plan, (documents, validation) in zip(sub_plans, results):
            statuses.append(validation.status)
            details.append({
                "index": sub_plan.index,
                "route_key": sub_plan.intent.route_key,
                "engine": sub_plan.engine,
                "status": validation.status,
                "detail": validation.detail,
                "retry_hint": validation.retry_hint,
                "doc_count": len(documents),
            })
            for doc in documents:
                key = (
                    doc.metadata.get("node_id"),
                    doc.metadata.get("chunk_id"),
                    doc.page_content[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                doc.metadata["sub_query_index"] = sub_plan.index
                doc.metadata["sub_query"] = sub_plan.query
                doc.metadata["sub_route_key"] = sub_plan.intent.route_key
                merged.append(doc)

        if not merged:
            return [], "missing", details
        if any(status == "missing" for status in statuses):
            return merged, "partial", details
        if any(status == "partial" for status in statuses):
            return merged, "partial", details
        return merged, "complete", details

    @staticmethod
    def _merge_documents(primary: List[Document], extra: List[Document]) -> List[Document]:
        """双 query 兜底合并；保留主结果顺序，去重后追加原始 query 的结果。"""
        merged = list(primary)
        seen = {
            (doc.metadata.get("node_id"), doc.metadata.get("chunk_id"), doc.page_content[:80])
            for doc in merged
        }
        for doc in extra:
            key = (
                doc.metadata.get("node_id"),
                doc.metadata.get("chunk_id"),
                doc.page_content[:80],
            )
            if key in seen:
                continue
            doc.metadata["dual_query"] = True
            merged.append(doc)
            seen.add(key)
        return merged

    def _execute_sub_plan(self, sub_plan: SubPlan) -> Tuple[List[Document], ValidationResult]:
        """执行一个子计划，并在证据不足时补一次文本检索。"""
        if sub_plan.engine == "none":
            return [], ValidationResult("complete", "不需要检索")

        try:
            documents = self.evidence_service.execute(sub_plan)
        except Exception as exc:
            logger.warning("证据检索失败 route=%s: %s", sub_plan.intent.route_key, exc)
            documents = []

        validation = self.validator.validate(sub_plan, documents)

        # 图谱/结构化证据缺失时，补一次现有 hybrid 检索（最多一次）
        if validation.status != "complete" and validation.retry_hint == "text":
            try:
                extra = self.text_adapter.search(sub_plan.query, sub_plan.top_k)
            except Exception as exc:
                logger.warning("文本补检索失败: %s", exc)
                extra = []
            if extra:
                documents = documents + extra
                validation = self.validator.validate(sub_plan, documents)

        return documents, validation

    def _fallback_search(self, query: str, top_k: int) -> List[Document]:
        """最终兜底：传统检索优先，其次返回空。"""
        if self.traditional_retrieval is None:
            return []
        try:
            return self.traditional_retrieval.hybrid_search(query, top_k)
        except Exception as exc:
            logger.error("传统检索兜底失败: %s", exc)
            return []

    # ------------------------------------------------------------------
    # 结果与分析组装
    # ------------------------------------------------------------------

    def _build_analysis(
        self,
        query: str,
        understanding: Understanding,
        plan: RoutePlan,
    ) -> QueryAnalysis:
        """把理解和计划压成兼容旧接口的 QueryAnalysis。"""
        domain = self.policy.domain
        default_spec = domain.spec(domain.default_route_key)
        top_intent = understanding.top_intent(default_spec.intent, default_spec.key)
        retrieval_plans = [item for item in plan.sub_plans if item.engine != "none"]
        primary = retrieval_plans[0] if retrieval_plans else (plan.sub_plans[0] if plan.sub_plans else None)
        primary_intent = primary.intent if primary else top_intent
        engine = primary.engine if primary else "none"

        complexity, relation, entities = self._analysis_metrics(understanding, plan, engine)
        reasoning = self._analysis_reasoning(primary_intent, engine, plan, understanding)

        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation,
            reasoning_required=bool(
                plan.is_multi_intent
                or any("REL" in item.evidence_profile for item in understanding.intents)
                or domain.spec(primary_intent.route_key).reasoning_required
            ),
            entity_count=len(entities),
            recommended_strategy=self._strategy_for_engine(engine),
            confidence=float(top_intent.confidence),
            reasoning=reasoning,
            standalone_query=understanding.standalone_query or query,
            needs_retrieval=plan.needs_retrieval,
            intent=primary_intent.intent,
            route_key=primary_intent.route_key,
            engine=engine,
            evidence_profile=list(primary_intent.evidence_profile),
            entities=entities,
            slots=dict(primary_intent.slots or {}),
            expected_source=list(primary_intent.expected_source),
            understanding=asdict(understanding),
            plan=plan,
            sub_queries=self._analysis_sub_queries(plan),
            is_multi_intent=plan.is_multi_intent,
            response_type=plan.response_type,
            clarify_question=plan.clarify_question,
            topic_id=self._topic_id(understanding, primary_intent, domain),
            used_history=understanding.context.used_history,
            topic_shift=understanding.context.topic_shift,
            disclaimer_required=understanding.disclaimer_required,
        )

    @staticmethod
    def _analysis_metrics(
        understanding: Understanding,
        plan: RoutePlan,
        engine: str,
    ) -> Tuple[float, float, List[str]]:
        """计算旧接口里的复杂度、关系强度和实体列表。"""
        raw = understanding.raw or {}
        complexity = raw.get("query_complexity")
        if not isinstance(complexity, (int, float)):
            complexity = min(
                1.0,
                0.2 + 0.2 * len(understanding.intents)
                + 0.1 * len({entity.name for item in understanding.intents for entity in item.entities})
                + (0.2 if plan.is_multi_intent else 0.0),
            )

        relation = raw.get("relationship_intensity")
        if not isinstance(relation, (int, float)):
            relation = 1.0 if (
                any("REL" in item.evidence_profile for item in understanding.intents)
                or engine == "graph_relation"
            ) else 0.5

        entities: List[str] = []
        for item in understanding.intents:
            for entity in item.entities:
                if entity.name not in entities:
                    entities.append(entity.name)

        return float(complexity), float(relation), entities

    @staticmethod
    def _analysis_reasoning(
        primary_intent,
        engine: str,
        plan: RoutePlan,
        understanding: Understanding,
    ) -> str:
        """生成兼容旧接口的一句话路由解释。"""
        reasoning = (
            f"intent={primary_intent.intent}, route_key={primary_intent.route_key}, "
            f"engine={engine}, source={primary_intent.expected_source}"
        )
        if plan.is_multi_intent:
            reasoning += f", 多意图={len(understanding.intents)}"
        return reasoning

    @staticmethod
    def _analysis_sub_queries(plan: RoutePlan) -> List[Dict[str, Any]]:
        """把子计划压成旧接口需要的字典列表。"""
        return [
            {
                "index": item.index,
                "intent": item.intent.intent,
                "route_key": item.intent.route_key,
                "query": item.query,
                "engine": item.engine,
                "confidence": item.intent.confidence,
                "entities": [
                    entity.name for entity in item.intent.entities if entity.name
                ],
                "depends_on": item.depends_on,
            }
            for item in plan.sub_plans
        ]

    @staticmethod
    def _strategy_for_engine(engine: str) -> SearchStrategy:
        """把新引擎名映射回旧的策略枚举。"""
        if engine in ("graph_entity", "graph_relation"):
            return SearchStrategy.GRAPH_RAG
        if engine in ("tips_pairing", "tips_safety", "tips_technique"):
            return SearchStrategy.TIPS_KNOWLEDGE
        if engine == "tips_rule":
            return SearchStrategy.RULE_ENGINE
        if engine == "none":
            return SearchStrategy.NO_RETRIEVAL
        if engine == "combined":
            return SearchStrategy.COMBINED
        return SearchStrategy.HYBRID_TRADITIONAL

    @staticmethod
    def _topic_id(
        understanding: Understanding,
        primary_intent,
        domain: DomainConfig,
    ) -> str:
        """缓存/会话主题键：当前菜名优先，其次只在“菜谱属性类”追问上沿用 active_recipe。"""
        context = understanding.context
        recipe_names = primary_intent.entity_names("recipe") if primary_intent else []
        if recipe_names:
            return recipe_names[0]

        route_key = primary_intent.route_key if primary_intent else ""
        spec = domain.spec(route_key)
        if spec.topic_mode == "chitchat":
            return context.active_recipe or route_key
        if spec.topic_mode == "pair":
            pair_names = [entity.name for entity in primary_intent.entities if entity.name]
            if len(pair_names) >= 2:
                return "+".join(pair_names[:2])

        if context.active_recipe and not context.topic_shift and spec.entity_first:
            return context.active_recipe
        return route_key or context.active_topic or "default"

    def _post_process_results(
        self,
        documents: List[Document],
        analysis: QueryAnalysis,
    ) -> List[Document]:
        """给结果补路由元数据；旧字段继续保留。"""
        for doc in documents:
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence,
                "route_key": analysis.route_key,
                "route_intent": analysis.intent,
                "route_engine": analysis.engine,
                "needs_retrieval": analysis.needs_retrieval,
            })
        return documents

    def _log_decision(
        self,
        query: str,
        analysis: QueryAnalysis,
        documents: List[Document],
        evidence_status: str,
        sub_evidence: List[Dict[str, Any]],
        started: float,
        session_id: str,
        error: str = "",
    ) -> None:
        """把一次路由决策写成一行 JSONL；落盘失败不影响主流程。"""
        understanding = analysis.understanding or {}
        sub_queries = [
            {
                "index": item.get("index"),
                "route_key": item.get("route_key", ""),
                "engine": item.get("engine", ""),
                "confidence": item.get("confidence"),
                "query": item.get("query", ""),
                "entities": item.get("entities", []),
            }
            for item in (analysis.sub_queries or [])
        ]
        self.decision_logger.log({
            "session_id": session_id,
            "query": query,
            "standalone_query": analysis.standalone_query,
            "route_key": analysis.route_key,
            "intent": analysis.intent,
            "engine": analysis.engine,
            "needs_retrieval": analysis.needs_retrieval,
            "response_type": analysis.response_type,
            "confidence": round(float(analysis.confidence), 4),
            "understanding_source": understanding.get("source", ""),
            "topic_id": analysis.topic_id,
            "used_history": analysis.used_history,
            "topic_shift": analysis.topic_shift,
            "is_multi_intent": analysis.is_multi_intent,
            "sub_queries": sub_queries,
            "evidence_status": evidence_status,
            "sub_evidence": sub_evidence,
            "doc_count": len(documents),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error": error,
        })

    def _ensure_entities(self) -> None:
        """首次查询时刷新实体词表（此时图谱数据已加载）。"""
        if self._entities_loaded:
            return
        self._entities_loaded = True
        try:
            self.understanding_service.set_known_entities(
                recipes=self.graph_adapter.get_recipe_names(),
                ingredients=self.graph_adapter.get_ingredient_names(),
                cuisines=self.graph_adapter.get_cuisines(),
            )
        except Exception as exc:
            logger.warning("实体词表加载失败: %s", exc)

    @staticmethod
    def _history_text(state: Dict[str, Any]) -> str:
        """只把“用户说过的话 + 约束”给理解层，不把 AI 答案拼进检索 query。"""
        if not state:
            return ""
        if state.get("history_text"):
            return str(state["history_text"])

        lines: List[str] = []
        recent = state.get("recent_user_queries") or []
        for item in recent[-2:]:
            lines.append(f"用户: {item}")
        constraints = state.get("constraints") or {}
        if constraints:
            lines.append(f"已知约束: {constraints}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 统计与解释
    # ------------------------------------------------------------------

    def _update_route_stats(self, strategy: SearchStrategy, engine: str = "") -> None:
        """更新路由统计；engine 优先，strategy 作为旧调用兜底。"""
        self.route_stats["total_queries"] += 1
        stat_key = ENGINE_STAT_KEYS.get(engine) or STRATEGY_STAT_KEYS.get(strategy)
        self.route_stats[stat_key or "no_retrieval_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息。"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats

        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total,
            "tips_ratio": self.route_stats["tips_count"] / total,
            "rule_ratio": self.route_stats["rule_count"] / total,
            "no_retrieval_ratio": self.route_stats["no_retrieval_count"] / total,
        }

    def _threshold_text(self, route_key: str) -> str:
        """把当前路由键的阈值格式化成一行，方便对照 confidence 看决策。"""
        threshold = self.policy._threshold(route_key)
        return (
            f"min_execute={threshold.min_execute:.2f}, "
            f"retrieval_override={threshold.retrieval_override:.2f}, "
            f"no_retrieval={threshold.no_retrieval:.2f}"
        )

    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程。"""
        analysis = self.analyze_query(query)

        lines = [
            "查询路由分析报告",
            "",
            f"查询：{query}",
            f"独立查询：{analysis.standalone_query}",
            "",
            "理解结果：",
            f"- 意图：{analysis.intent}",
            f"- 路由键：{analysis.route_key}",
            f"- 证据画像：{'/'.join(analysis.evidence_profile) or '无'}",
            f"- 需要检索：{'是' if analysis.needs_retrieval else '否'}",
            f"- 实体：{', '.join(analysis.entities) or '无'}",
            f"- 槽位：{analysis.slots or '无'}",
            f"- 置信度：{analysis.confidence:.2f}",
            f"- 阈值：{self._threshold_text(analysis.route_key)}",
            "",
            "执行计划：",
        ]

        if analysis.plan and analysis.plan.sub_plans:
            for item in analysis.plan.sub_plans:
                lines.append(
                    f"- [{item.intent.route_key}] engine={item.engine}, "
                    f"confidence={item.intent.confidence:.2f} "
                    f"({self._threshold_text(item.intent.route_key)}), "
                    f"top_k={item.top_k}, validator={item.validator}, query={item.query}"
                )
        else:
            lines.append("- 不检索" + (f"（{analysis.clarify_question}）" if analysis.clarify_question else ""))

        lines.extend([
            "",
            f"旧版特征：复杂度 {analysis.query_complexity:.2f}，"
            f"关系密集度 {analysis.relationship_intensity:.2f}，"
            f"推理需求 {'是' if analysis.reasoning_required else '否'}",
            f"决策理由：{analysis.reasoning}",
        ])
        return "\n".join(lines)
