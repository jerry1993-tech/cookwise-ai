"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配。

预览：
本模块按单一职责拆成若干能力组件，由末尾的 `GraphRAGRetrieval` 门面组合（多继承）而成。
对外接口与拆分前完全一致，仍主要提供三个「入口」方法：
1. `initialize()`：连接 Neo4j 并预热图索引（后续所有图操作都靠它建立的 `self.driver`）。
2. `graph_rag_search(query, top_k)`：主搜索接口。内部完整走一遍「理解意图 → 按类型分发 → 图结构排序」的流程，最后返回一批 `Document` 对象。
3. `understand_graph_query(query)`：单独把「意图理解」这一步暴露出来，方便测试和路由器复用。

能力组件划分（每个类只负责一件事，方法体与拆分前逐字一致）：
- `GraphStoreMixin`：连接生命周期、只读 Cypher、节点/关系序列化、度数补齐、身份键与关系去重。
- `GraphIndexMixin`：图索引预热、实体名规范化、同族别名对齐、候选实体抽取、种子节点解析。
- `QueryUnderstandingMixin`：自然语言 → `GraphQuery`（LLM + 本地降级）、复杂度分析与自适应查询规划。
- `PathModelMixin`：`GraphPath` 的构建、评分、搭配证据读取、去重与 Neo4j 记录解析。
- `GraphTraversalMixin`：多跳遍历主引擎（深度/种子收口、变长路径、搭配共现通道）。
- `PathQueryMixin`：实体关系、最短路径、相似实体三条定向查询策略。
- `SubgraphExtractionMixin`：两阶段知识子图抽取、行组装、缓存淘汰与三级降级。
- `GraphReasoningMixin`：识别子图拓扑模式，构造并校验可溯源的推理链。
- `DocumentFormattingMixin`：把路径/子图渲染为自然语言描述与 `Document`，并计算图结构分。
- `ResultRankingMixin`：词法 + 图结构 + 类型先验 + 可溯源程度的四分量融合排序。
- `GraphRAGRetrieval`：门面，持有共享状态并串联主检索流程；子类可覆盖任一组件实现替换（热插拔）。

组件之间不持有彼此的实例：共享状态（driver / 各类缓存 / 索引）统一由门面在 `__init__` 中建立，
各方法只通过 `self` 协作。因此组件的依赖面就是门面暴露的这几个状态与同职责方法，
新增或替换一个组件时不需要改其他组件，也不需要改上层调用方。

===============================================================================
实现说明（写给维护者）
===============================================================================
本模块面向的真实图谱是「中式烹饪知识图谱」，其结构经过实测确认，若干特性直接决定了这里的算法取舍，
把这些前提写下来，是为了避免后续维护者按「常识」想当然地改坏它：

1. 【食材是按菜谱私有的】Ingredient 节点 2906 个，而 REQUIRES 边仅 2905 条，且每个食材节点度数几乎恒为 3
   （REQUIRES + BELONGS_TO_CATEGORY + HAS_CONCEPT_TYPE）。也就是说**不存在跨菜谱共享的食材节点**，
   「食材 → 菜品 → 同类食材」这种常见假设在本图谱上【走不通】。
   菜谱之间真正可用的连通通路只有：
       Recipe -[:BELONGS_TO_CATEGORY]-> Category <-[:BELONGS_TO_CATEGORY]- Recipe   （同分类，最主要）
       Recipe -[:SIMILAR]- Recipe                                                     （稀疏，仅约 46 道菜有）
       Recipe -[:CONTAINS_STEP]-> CookingStep -[:USES_SAME_METHOD]-(...)- Recipe      （同技法，精细）
       以及 cuisineType / difficulty 这类属性级同组关系
2. 【hub 节点必须剪枝】ConceptType 'Ingredient' 度数 2906、Category '调料' 1556、CookingStep '步骤2' 511。
   不做度数剪枝的话，度数 3 的变长路径实测会产生 127214 条路径。
3. 【变长路径与 shortestPath 的上界不能参数化】Neo4j 明确报错：「Parameter maps cannot be used in MATCH patterns」。
   因此深度只能以【经过钳制的整数】内插进 Cypher 字符串（见 `_MAX_TRAVERSAL_DEPTH` 与 `_clamp_int`），
   这正是本模块所有深度参数都强制走 `_clamp_int` 的原因。
4. 【关系类型在 Python 侧只能用 rel.type 取】Cypher 里的 `type(rel)` 才是类型；而 neo4j 驱动的 Relationship 对象
   上是 `rel.type` 属性，`type(rel).__name__` 只会得到字符串 'Relationship'（这是本模块修掉的一个真实缺陷）。
5. 【脏数据】图谱里混有 GraphStatistics / RelationshipStatistics / Root / CookingMethod / CookingTool 等统计与层次节点，
   以及 nodeId 为 '200000000'（name='菜谱'）、'300000000'（name='食材'）的汇总占位节点。
   所有查询都通过标签白名单 + nodeId 黑名单过滤（见 `_MEANINGFUL_LABELS` / `_PLACEHOLDER_NODE_IDS`）。
6. 【LLM 是推理模型】本项目的 deepseek-flash 会先输出 reasoning_content，若不关闭思考，max_tokens 会被推理过程吃光，
   返回的 content 为空字符串。因此所有 LLM 调用都必须带 `extra_body={"thinking": {"type": "disabled"}}`，
   并且对返回内容做「剥代码围栏 + 容错 JSON 解析 + 空内容判定」（见 `_call_llm_json` / `_extract_json_object`）。
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

from .graph_rag_retrieval_components.document_formatting_and_ranking import (
    DocumentFormattingMixin,
    ResultRankingMixin,
)
from .graph_rag_retrieval_components.entity_name_index import GraphIndexMixin
from .graph_rag_retrieval_components.neo4j_store_access import GraphStoreMixin
from .graph_rag_retrieval_components.path_retrieval_engine import (
    GraphTraversalMixin,
    PathModelMixin,
    PathQueryMixin,
)
from .graph_rag_retrieval_components.query_understanding import QueryUnderstandingMixin
from .graph_rag_retrieval_components.subgraph_extraction_and_reasoning import (
    GraphReasoningMixin,
    SubgraphExtractionMixin,
)
from .graph_rag_retrieval_components.shared_constants_helpers_models import (
    GraphPath,
    GraphQuery,
    KnowledgeSubgraph,
    QueryType,
    _MAX_TRAVERSAL_DEPTH,
    _coerce_query_text,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GraphRAGRetrieval",
    "QueryType",
    "GraphQuery",
    "GraphPath",
    "KnowledgeSubgraph",
]

class GraphRAGRetrieval(
    GraphStoreMixin,
    GraphIndexMixin,
    QueryUnderstandingMixin,
    PathModelMixin,
    GraphTraversalMixin,
    PathQueryMixin,
    SubgraphExtractionMixin,
    GraphReasoningMixin,
    DocumentFormattingMixin,
    ResultRankingMixin,
):
    """真正的图RAG检索系统（门面）。

    对外接口与拆分前完全一致：initialize / graph_rag_search / understand_graph_query /
    multi_hop_traversal / extract_knowledge_subgraph / graph_structure_reasoning /
    adaptive_query_planning / close，以及原有全部下划线辅助方法（测试与排障直接调用）。
    内部按单一职责拆成十个能力组件；子类覆盖任一组件的方法即可替换实现，调用方无需修改。

    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

        # 意图理解结果缓存：key = 原始查询串，value = GraphQuery。
        # 同一个问题常被路由器与生成阶段反复问到，缓存可省掉重复的 LLM 调用。
        self._query_understanding_cache: Dict[str, "GraphQuery"] = {}

        # 实体名 -> nodeId 列表 的反向索引，由 _build_graph_index 预热，用于把自然语言里的实体名快速落到图谱节点上
        self._name_index: Dict[str, List[str]] = defaultdict(list)

        # 【属性值词表】菜系（Recipe.cuisineType）、分类（category）这类【只作为属性存在】的值。
        # 它们在图谱里没有同名节点（实测「川菜」不存在同名节点，只存在于 33 个 Recipe 的 cuisineType），
        # 因此不在 _name_index 里；但 _resolve_seed_nodes 的属性分支【能】匹配到它们。
        # 单独一张表而不是混进 _name_index：后者的契约是"名称 -> nodeId"，混入属性值会污染别名定位。
        self._vocab_phrases: Set[str] = set()

        # 子图缓存的插入顺序，用于容量超限时做 FIFO 淘汰（无需引入额外依赖）
        self._subgraph_cache_order: deque = deque()

        # 最近一次多跳扩展【实际使用】的深度（由 _expand_multi_hop_paths 回填）。
        # 请求深度与实际深度可能不同（种子度数会让深度降级），而"兜底查询是否与主路径等价"
        # 必须按实际深度判断，否则会原样重跑一遍刚跑过的查询。
        self._last_effective_depth: Optional[int] = None

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")

        # 【入口防御一：query 必须是字符串】
        # understand_graph_query 内部会 query.strip()，而它是在下面的 try 之前调用的，
        # int / list / dict / float 会抛 AttributeError 直接穿透公开入口（实测均复现）。
        # 检索模块对上层应当是"降级不抛错"的语义，这里统一收口为字符串。
        query = _coerce_query_text(query)

        # 【入口防御二：top_k 必须钳制成整数】
        # 此前零校验，直接 results[:top_k]：
        #   - top_k=None  → 切片键退化为 None，完全不截断，返回底层全部结果（实测 60 条）；
        #   - top_k 为负  → 按尾部截断，返回近全量（-1 → 59 条）；
        #   - 字符串/浮点 → 切片抛 TypeError，被下面的兜底 except 吞成 return []，
        #     把"参数类型错误"伪装成"图里没数据"，调用方无法区分。
        # 非整数输入记一条 warning 后回退到默认值，越界值钳到 [1, ...) ——
        # top_k<=0 对检索 API 没有意义，返回 0 条只会让调用方以为"图里没有证据"。
        try:
            top_k = int(top_k)
        except (TypeError, ValueError, OverflowError):
            logger.warning(f"top_k 非法（{top_k!r}），回退为默认值 5")
            top_k = 5
        if top_k < 1:
            logger.warning(f"top_k={top_k} 无意义（<=0），钳制为 1")
            top_k = 1

        if not self._is_driver_ready():
            logger.warning("Neo4j连接未建立，返回空结果")
            return []

        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")

        results = []

        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.SUBGRAPH:
                # 子图提取
                subgraph = self.extract_knowledge_subgraph(graph_query)

                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)

                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))

            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.CLUSTERING:
                # 聚类相似查询：走相似实体检索，产出"以相似项为成员"的路径型证据
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            # 2.5 结果回收：若主策略没拿到证据，退一步用子图策略补一次，
            #     避免「路径分支恰好为空」时对用户静默返回空结果
            if not results:
                logger.info("主策略未取得结果，回退到子图检索")
                subgraph_query = GraphQuery(
                    query_type=QueryType.SUBGRAPH,
                    source_entities=graph_query.source_entities,
                    target_entities=graph_query.target_entities,
                    max_depth=min(graph_query.max_depth, _MAX_TRAVERSAL_DEPTH),
                    max_nodes=graph_query.max_nodes,
                )
                subgraph = self.extract_knowledge_subgraph(subgraph_query)
                if subgraph.central_nodes or subgraph.connected_nodes:
                    reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                    results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))

            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)

            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]

        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []
