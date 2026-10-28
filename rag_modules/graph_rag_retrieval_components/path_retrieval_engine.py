"""路径检索引擎：GraphPath 构建与评分、多跳遍历、实体关系 / 最短路径 / 相似实体三条定向查询。"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .shared_constants_helpers_models import (
    GraphPath,
    GraphQuery,
    QueryType,
    _DEEP_SEED_DEGREE_LIMIT,
    _DEFAULT_RELATION_WEIGHT,
    _HUB_DEGREE_THRESHOLD,
    _MAX_DEEP_SEED_COUNT,
    _MAX_PATH_RESULTS,
    _MAX_SUBGRAPH_NODES,
    _MAX_TRAVERSAL_DEPTH,
    _PAIRING_BUDGET_RATIO,
    _PAIRING_MIN_RESULTS,
    _PAIRING_ROW_FACTOR,
    _PAIRING_SUPPORT_BONUS,
    _PAIRING_SUPPORT_SATURATION,
    _PAIRING_WITNESS_LIMIT,
    _PATH_LABEL_FILTER_X,
    _PLACEHOLDER_NODE_IDS,
    _RELATION_WEIGHTS,
    _clamp,
    _dedup_preserve_order,
)

logger = logging.getLogger(__name__)


class PathModelMixin:
    """路径模型：GraphPath 的构建、相关性评分、搭配证据读取、去重与 Neo4j 记录解析。"""

    def _score_path(self, path: GraphPath) -> float:
        """给一条图路径打相关性分。

        三个分量（权重合计 1.0）：
          - 关系语义强度 (0.5)：路径上各关系类型权重的均值，体现「这条路径说的话是否靠谱」；
          - 路径紧凑度 (0.3)：1/(1+跳数)，越短越可信（间接关联噪声更大）；
          - 节点信息量 (0.2)：路径节点度数的归一化均值，避免全是孤立叶子节点的路径。
        """
        if not path.relationships:
            rel_strength = _DEFAULT_RELATION_WEIGHT
        else:
            weights = [
                _RELATION_WEIGHTS.get(str(r.get("type") or ""), _DEFAULT_RELATION_WEIGHT)
                for r in path.relationships
            ]
            rel_strength = sum(weights) / float(len(weights))

        compactness = 1.0 / (1.0 + max(int(path.path_length or 1), 1))

        if path.nodes:
            degree_score = sum(min(int(n.get("degree") or 0) / 50.0, 1.0) for n in path.nodes) / float(len(path.nodes))
        else:
            degree_score = 0.0

        score = 0.5 * rel_strength + 0.3 * compactness + 0.2 * degree_score
        return round(float(score), 6)

    def _pairing_support(self, path: GraphPath) -> Tuple[int, int]:
        """读取路径上的搭配共现证据。

        证据写在【关系字典】上（见 _expand_pairing_paths）：pairing_support = 共现菜谱数，
        pairing_source_names = 该搭配覆盖的同族食材名数。
        一条路径上可能有多条关系带该字段，取最大值；任何异常（类型不符）都按"无证据"处理，
        因为这里只是加分/展示的依据，不值得为脏数据中断检索。

        Returns:
            (共现菜谱数, 同族食材名数)；无证据时为 (0, 0)
        """
        support, src_count = 0, 0
        for rel in (path.relationships or []):
            try:
                support = max(support, int(rel.get("pairing_support") or 0))
                src_count = max(src_count, int(rel.get("pairing_source_names") or 0))
            except (TypeError, ValueError):
                continue
        return support, src_count

    def _pairing_support_bonus(self, path: GraphPath) -> float:
        """把搭配共现强度折成一个【有界】的结构证据加成，取值 [0, _PAIRING_SUPPORT_BONUS]。

        为什么是"结构证据"而不是词法证据：结构完全相同的 食材-菜谱-食材 路径（2 跳、两条 REQUIRES）
        在 _score_path / _path_struct_score 里拿到的是【完全相同】的分数，
        唯一能区分"鸡肉配姜很常见"和"鸡肉配欧芹只出现一次"的信号就是共现菜谱数，
        它衡量的是图谱里这条关联的强度，属于结构侧。
        上限存在的原因：避免高频搭配压过一切其它证据，也让同分时退回稳定排序的输入顺序。
        """
        support, _ = self._pairing_support(path)
        if support <= 0:
            return 0.0
        return _PAIRING_SUPPORT_BONUS * min(float(support) / _PAIRING_SUPPORT_SATURATION, 1.0)

    def _build_graph_path(self, nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]],
                          path_type: str, path_length: Optional[int] = None) -> Optional[GraphPath]:
        """把节点/关系字典组装成 GraphPath，并做统一的脏数据过滤。

        返回 None 表示这条路径不该进入结果集（节点为空、或混入了统计/占位脏节点）。
        """
        # 过滤掉统计/层次节点与占位节点；只要混入一个，整条路径即为脏路径
        clean_nodes = [n for n in nodes if self._is_meaningful_node(n)]
        if not clean_nodes or len(clean_nodes) != len(nodes):
            return None

        if len(clean_nodes) == 1 and not rels:
            # 单节点无关系无法构成路径
            return None

        return GraphPath(
            nodes=clean_nodes,
            relationships=list(rels or []),
            path_length=int(path_length if path_length is not None else max(len(clean_nodes) - 1, 1)),
            relevance_score=0.0,   # 由 _score_path 统一回填，保证评分口径一致
            path_type=path_type,
        )

    def _dedup_paths(self, paths: List[GraphPath]) -> List[GraphPath]:
        """按「节点序列」去重，保留相关性最高的那条。

        Cypher 的无向多跳匹配会对同一批节点产出方向/顺序不同的重复路径，必须去重否则 top_k 会被挤占。
        """
        best: Dict[Tuple[str, ...], GraphPath] = {}
        for path in paths:
            if not path or not path.nodes:
                continue
            key = tuple(str(n.get("id") or n.get("name") or "") for n in path.nodes)
            if key not in best or path.relevance_score > best[key].relevance_score:
                best[key] = path
        ordered = sorted(best.values(), key=lambda p: p.relevance_score, reverse=True)
        return ordered[:_MAX_PATH_RESULTS]

    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录。

        支持两种形态：
          - 含 "path" 键的 neo4j Path 对象（推荐形态，能拿到真实方向与关系类型）；
          - 早期版本的 path_nodes / rels / path_len / relevance 扁平键（保留兼容）。
        关系类型一律通过 _rel_to_dict 读取（内部使用驱动的 rel.type），
        避免早期 `type(rel).__name__` 只得到 'Relationship' 的缺陷。
        """
        try:
            if record.get("path") is not None:
                neo_path = record["path"]
                nodes = [self._node_to_dict(n) for n in neo_path.nodes]
                rels = [self._rel_to_dict(r) for r in neo_path.relationships]
                length = len(rels)
                relevance = record.get("relevance")
                # 查询里带了 relevance（例如全文/向量侧已经打过分的图路径）就直接沿用，
                # 否则交回 _score_path 统一评分，保证同一批结果的分数量纲一致
                if isinstance(relevance, (int, float)):
                    return GraphPath(
                        nodes=nodes,
                        relationships=rels,
                        path_length=length,
                        relevance_score=float(relevance),
                        path_type="multi_hop",
                    )
                built = self._build_graph_path(
                    nodes=nodes, rels=rels, path_type="multi_hop",
                    path_length=length if length else None,
                )
                if built is not None:
                    built.relevance_score = self._score_path(built)
                return built

            # 兼容路径：扁平键形态
            path_nodes = []
            for node in (record.get("path_nodes") or []):
                if isinstance(node, dict):
                    path_nodes.append(node)
                else:
                    path_nodes.append(self._node_to_dict(node))

            relationships = []
            for rel in (record.get("rels") or []):
                if isinstance(rel, dict):
                    relationships.append(rel)
                elif isinstance(rel, list):
                    relationships.extend([self._rel_to_dict(item) for item in rel])
                else:
                    relationships.append(self._rel_to_dict(rel))

            relevance = record.get("relevance")
            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=int(record.get("path_len") or max(len(path_nodes) - 1, 1)),
                relevance_score=float(relevance) if isinstance(relevance, (int, float)) else 0.0,
                path_type="multi_hop"
            )

        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None

class GraphTraversalMixin:
    """多跳遍历：按查询类型分发，负责深度/种子收口、通用变长路径与搭配共现通道。"""

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳图遍历：这是图RAG的核心优势
        通过图结构发现隐含的知识关联
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")

        paths: List[GraphPath] = []

        if not self._is_driver_ready():
            logger.error("Neo4j连接未建立")
            return paths

        depth = _clamp(getattr(graph_query, "max_depth", 2), 1, _MAX_TRAVERSAL_DEPTH, 2)
        query_type = getattr(graph_query, "query_type", QueryType.SUBGRAPH)

        # 主路径实际使用的深度由 _expand_multi_hop_paths 回填（它内部会按种子度数降级）。
        # 先置 None：避免上一次查询的残留值影响本次的"兜底是否等价"判定。
        self._last_effective_depth = None

        try:
            # 按查询类型分派到不同的遍历策略；每条策略内部自行完成种子解析与结果组装
            with self.driver.session() as session:
                if query_type == QueryType.ENTITY_RELATION:
                    paths = self._find_entity_relations(graph_query, session)
                elif query_type == QueryType.PATH_FINDING:
                    paths = self._find_shortest_paths(graph_query, session)
                elif query_type == QueryType.CLUSTERING:
                    paths = self._find_similar_entities(graph_query, session)
                else:
                    # MULTI_HOP / SUBGRAPH：以种子为起点做受限多跳扩展
                    paths = self._expand_multi_hop_paths(graph_query, session, depth)
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
            paths = []

        # 兜底：按类型分派后若为空（例如 PATH_FINDING 没解析出目标实体、聚类种子无相似项），
        # 退一步做一次通用的邻居扩展，尽量避免对用户静默返回空结果。
        # 【但 MULTI_HOP / SUBGRAPH 分支不需要这个兜底】它们在主路径上调用的就是
        # _expand_multi_hop_paths，且实际深度 <= 2 时兜底参数 min(depth,2)==实际深度，两轮调用的
        # Cypher 文本与参数【全等】（实测一次调用共发出 8 条 Cypher）；种子解析又是确定性的，
        # 变长路径集在 hub 剪枝下单调，所以第二轮必然得到同样的空结果 ——
        # 纯粹是把最贵的那条查询原样重跑一遍，耗时翻倍（实测 SUBGRAPH d=2 约 2.0x）。
        # 只有 ENTITY_RELATION / PATH_FINDING / CLUSTERING 三类分支，主路径用的是别的查询，
        # 兜底才可能真的补到结果。
        #
        # 【判据必须用"实际深度"而不是"请求深度"】_expand_multi_hop_paths 内部会按种子度数降级
        # （hub 种子 3->1，高密度种子 3->2）。用请求深度判断时，max_depth=3 + hub 种子会出现：
        # 主路径按请求 3 判为"非等价"→ 放行兜底 → 兜底以 min(3,2)=2 进入、同样被降级到 1
        # → 发出的 Cypher 与参数与主路径【逐字相同】，白跑一遍。
        # 用实际深度判断则天然覆盖：实际深度 <= 2 时，兜底的 2/1 跳与它等价（降级规则是单调的，
        # 实际深度 1 意味着种子是 hub，兜底也必然降到 1），故一律跳过。
        effective_depth = (self._last_effective_depth
                           if self._last_effective_depth is not None else depth)
        needs_generic_fallback = (
            query_type in (QueryType.ENTITY_RELATION, QueryType.PATH_FINDING, QueryType.CLUSTERING)
            or effective_depth > 2
        )
        if not paths and needs_generic_fallback:
            logger.info("按查询类型未取得路径，执行通用邻居扩展兜底")
            try:
                with self.driver.session() as session:
                    paths = self._expand_multi_hop_paths(graph_query, session, min(depth, 2))
            except Exception as e:
                logger.warning(f"通用多跳兜底失败: {e}")
        elif not paths:
            logger.info("主路径未取得路径，且兜底查询与主路径完全等价，跳过一次重复遍历")

        paths = self._dedup_paths(paths)
        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths

    def _effective_depth_for_seeds(self, seeds: List[Dict[str, Any]], depth: int) -> int:
        """按种子节点的度数决定实际可用的遍历深度。

        背景（实测）：hub 剪枝只作用于路径的中间节点，种子作为端点从未受检。而图谱里存在
        度数 511 的 CookingStep（大量同名「步骤2/步骤3」节点）与度数 1556 的 Category。
        若把这类节点当种子做 3 跳变长路径展开，Neo4j 会枚举上百万行后オ再被 LIMIT 截断
        （实测 dbHits=12,809,992、枚举 1,395,178 行、6.7-9.7s），而 depth=2 时同一查询仅 296ms。
        换言之 LIMIT 只能截断「返回」，截断不了「枚举」。

        因此这里分成两档（都是"用可接受的精度换掉不可接受的延迟"）：
          - 度数 > _HUB_DEGREE_THRESHOLD：压到 1（只取直接邻居）；
          - 度数 > _DEEP_SEED_DEGREE_LIMIT 且请求了 3 跳：压到 2。
            3 跳枚举量随种子度数相乘增长，实测度数 83 的种子 3 跳要 4.7s 却只比 2 跳多 2 个节点，
            而降级到 2 跳只要 85ms；度数 <= 12 的小种子（含全部食材种子，度数恒为 3）不受影响，
            它们才是"第三跳真能走到新内容"的那一类（实测新增 32 个节点）。
        退化只发生在高密度种子上，以菜谱/食材为种子的常规查询不受影响。
        """
        if not seeds:
            return depth
        max_degree = max(int(s.get("degree") or 0) for s in seeds)
        if max_degree > _HUB_DEGREE_THRESHOLD and depth > 1:
            logger.info(
                f"种子节点度数为 {max_degree}（> {_HUB_DEGREE_THRESHOLD}），"
                f"遍历深度由 {depth} 降级为 1 以避免变长路径枚举爆炸"
            )
            return 1
        if depth > 2 and max_degree > _DEEP_SEED_DEGREE_LIMIT:
            logger.info(
                f"种子节点度数为 {max_degree}（> {_DEEP_SEED_DEGREE_LIMIT}），"
                f"遍历深度由 {depth} 降级为 2：该种子的第 3 跳只是在已覆盖邻域内绕圈，"
                f"枚举量却随度数相乘增长"
            )
            return 2
        return depth

    def _select_expansion_seeds(self, seeds: List[Dict[str, Any]], depth: int) -> List[str]:
        """挑选真正参与展开的种子 id。

        Cypher 里用的是同一条变长路径模式，UNWIND 多个种子意味着【每个种子各自完整枚举一遍】，
        而 LIMIT 作用在并集上、只能截断返回行数，截断不了枚举量。实测：
        1 个菜谱种子 145ms / 5 个种子 4,394ms，而两者返回的都是那 60 条路径。
        所以深度 >= 2 时限制种子个数（浅遍历便宜，可全用）。

        seeds 已由 _resolve_seed_nodes 按 (匹配级别 DESC, 度数 DESC) 排好序，
        因此取前 N 个即"最匹配且信息量最大"的那几个。
        """
        if not seeds:
            return []
        limit = len(seeds) if depth <= 1 else _MAX_DEEP_SEED_COUNT
        selected = [s["id"] for s in seeds[:limit] if s.get("id")]
        if len(seeds) > len(selected):
            logger.info(f"深遍历种子数由 {len(seeds)} 收敛为 {len(selected)}（其余种子成本线性叠加但结果高度重叠）")
        return selected

    def _expand_multi_hop_paths(self, graph_query: GraphQuery, session,
                                depth: int) -> List[GraphPath]:
        """受限多跳扩展：本模块多跳检索的主力实现。

        为什么不用「一次性把 N 跳全查出来」：
        实测从单个菜谱出发的变长路径 1 跳 35 条、2 跳 698 条、3 跳【127214 条】，
        因此这里做了五层收口：
          1. 深度钳制在 [_MAX_TRAVERSAL_DEPTH] 以内，并以字面量内插（Neo4j 不允许参数化上界）；
          2. Cypher 内过滤标签白名单与占位脏节点，把统计/层次节点排除在路径之外；
          3. Cypher 内 LIMIT 控制单次返回规模，再在 Python 侧做 hub 剪枝（度数超阈值的中间节点剔除），
             因为把 COUNT{(n)--()} 放进路径匹配里逐节点求值代价过高，不如取回后批量补度数再剪；
          4. 【种子侧也要收口】：hub 剪枝原本只作用于路径的"中间节点"，而种子是端点、从不被检查。
             实测把度数 511 的 CookingStep '步骤3'（图谱里存在大量同名步骤节点）当种子做 3 跳展开，
             VarLengthExpand 会枚举 1,395,178 行、耗时 6.7-9.7s，而 LIMIT 60 只能截断【返回】不能截断【枚举】。
             所以这里按种子的最大度数主动降级深度（见 _effective_depth_for_seeds），
             并限制参与深展开的种子个数（见 _MAX_DEEP_SEED_COUNT）——多种子是线性叠加的枚举成本，
             实测 5 个种子做 3 跳展开要 4.4s，而结果同样是那 60 条。
          5. 目标实体约束走「名称/标签/分类/菜系」多字段语义匹配，而不是错误的 labels() 直比。
          6. 【本方法有两条通道，共用一份路径预算】：
             - 搭配共现通道（_expand_pairing_paths）：把"X 配什么 Y"这类问句解释为
               「X 与 Y 经由同一个菜谱共现」，直接用 食材-菜谱-食材 的结构化两跳回答，
               并按共现菜谱数排序。它按【名字】锚定源侧，因此同名反规范化副本全部参与，
               不受 _MAX_DEEP_SEED_COUNT 限制。
             - 通用变长路径通道（本方法主体）：回答"通过什么连到"这类开放问句。
             预算分配是【搭配通道优先】：通用通道的 LIMIT 是"最短路径优先"，
             1 跳路径（食材→菜谱）会把 LIMIT 全部吃掉，2 跳的"食材→菜谱→食材"根本排不进来。
             实测「鸡肉配什么蔬菜好？」改造前只返回 2 条路径（都来自同一个菜谱），
             搭配通道介入后返回 27 种蔬菜。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=5, session=session)
        if not seeds:
            logger.info(f"多跳扩展未解析出种子节点: {graph_query.source_entities}")
            return []

        # 种子侧的两道收口（顺序重要：先按度数决定深度，再按深度决定用几个种子）
        depth = self._effective_depth_for_seeds(seeds, depth)
        # 回填"本次实际使用的深度"：调用方（multi_hop_traversal）靠它判断兜底查询是否与主路径等价
        self._last_effective_depth = depth
        seed_ids = self._select_expansion_seeds(seeds, depth)
        target_keywords = self._normalize_target_keywords(graph_query.target_entities)
        rel_types = [r for r in (graph_query.relation_types or []) if r in _RELATION_WEIGHTS]
        effective_rel_types = rel_types or list(_RELATION_WEIGHTS.keys())

        # 【搭配共现通道】先跑，并且它的产出【先从总预算里扣除】（见预算分配的注释）。
        # 触发条件三选全满足才跑：
        #   - depth >= 2：1 跳只能答"这个食材属于哪个类目"，答不了"和谁搭配"；
        #   - target_keywords 非空：搭配通道的目标侧收敛完全依赖关键词，没有关键词就退化成全量共现；
        #   - 种子里有 Ingredient：源侧要按名字锚定，全是菜谱/步骤种子时这条通道无意义。
        pairing_paths: List[GraphPath] = []
        if depth >= 2 and target_keywords and any(
                "Ingredient" in (s.get("labels") or []) for s in seeds):
            pairing_paths = self._expand_pairing_paths(
                graph_query, session, seeds, target_keywords,
                budget=max(_PAIRING_MIN_RESULTS,
                           int(_MAX_PATH_RESULTS * _PAIRING_BUDGET_RATIO)),
            )
            if len(pairing_paths) < _PAIRING_MIN_RESULTS:
                # 命中太少说明这条问句不是"搭配"型（或图谱里确实没有共现），
                # 与其塞几条弱证据进去污染排序，不如整条通道退场、把预算还给通用通道。
                logger.info(f"搭配共现命中 {len(pairing_paths)} 条，低于阈值 {_PAIRING_MIN_RESULTS}，放弃该通道")
                pairing_paths = []

        # 预算分配：搭配路径【优先占用】总预算，通用变长路径只拿剩下的部分。
        # 为什么必须让路：通用通道的 LIMIT 是"最短路径优先"（ORDER BY path_len ASC），
        # 1 跳路径（食材->菜谱）会把 LIMIT 全部吃掉，2 跳的"食材->菜谱->食材"根本排不进来 ——
        # 这正是改造前「鸡肉配什么蔬菜」只返回 2 条路径的直接原因。
        # 剩额为 0 时干脆不跑通用查询：省掉一次 680ms 级的枚举，且不会产出任何会被丢弃的结果。
        generic_limit = max(_MAX_PATH_RESULTS - len(pairing_paths), 0)
        if generic_limit == 0:
            logger.info(f"搭配共现已占满路径预算（{len(pairing_paths)} 条），跳过通用变长路径查询")
            pairing_paths.sort(key=lambda p: p.relevance_score, reverse=True)
            return pairing_paths

        # 深度以字面量内插：Neo4j 明确拒绝 `[*1..$depth]` 这种参数化上界，
        # 而 depth 已由 _clamp 与 _effective_depth_for_seeds 保证是 [1,3] 内的整数，不存在注入风险。
        cypher = f"""
        MATCH (source) WHERE source.nodeId IN $seed_ids
        MATCH path = (source)-[*1..{depth}]-(target)
        WHERE NOT target = source
          AND all(x IN nodes(path) WHERE ({_PATH_LABEL_FILTER_X}))
          AND all(x IN nodes(path) WHERE NOT coalesce(x.nodeId, '') IN $placeholder_ids)
          AND any(r IN relationships(path) WHERE type(r) IN $rel_types)
          AND ($target_keywords = []
               OR any(k IN $target_keywords WHERE
                    toLower(coalesce(target.name, '')) CONTAINS k
                    OR toLower(coalesce(target.category, '')) CONTAINS k
                    OR toLower(coalesce(target.cuisineType, '')) CONTAINS k
                    OR any(lbl IN labels(target) WHERE toLower(lbl) CONTAINS k)))
        WITH path, length(path) AS path_len, nodes(path) AS pnodes, relationships(path) AS prels
        ORDER BY path_len ASC
        LIMIT $limit
        RETURN pnodes, prels, path_len
        """

        rows = []
        try:
            rows = [dict(record) for record in session.run(cypher, {
                "seed_ids": seed_ids,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "rel_types": effective_rel_types,
                "target_keywords": target_keywords,
                "limit": generic_limit,
            })]
        except Exception as e:
            # 通用通道失败不该连带丢掉已经拿到的搭配路径：两者是独立证据源，失败隔离。
            logger.error(f"多跳路径查询失败: {e}")
            return pairing_paths

        paths: List[GraphPath] = []
        for row in rows:
            path = self._build_graph_path(
                nodes=[self._node_to_dict(n) for n in (row.get("pnodes") or [])],
                rels=[self._rel_to_dict(r) for r in (row.get("prels") or [])],
                path_type="multi_hop",
            )
            if path:
                paths.append(path)

        if not paths:
            return pairing_paths

        # 统一补齐度数后再剪枝：hub 剪枝与结构评分都依赖度数
        all_nodes = [n for p in paths for n in p.nodes]
        self._attach_degrees(all_nodes)

        pruned: List[GraphPath] = []
        dropped_hub = 0
        for path in paths:
            # 中间节点（非起点、非终点）若为 hub 则整条路径丢弃：
            # 实测 Category '调料'(1556) 这类 hub 会把任意两个菜谱连起来，属于噪声而非知识
            intermediates = path.nodes[1:-1]
            if any(self._is_hub(n) for n in intermediates):
                dropped_hub += 1
                continue
            path.relevance_score = self._score_path(path)
            pruned.append(path)

        if dropped_hub:
            logger.info(f"hub 剪枝丢弃 {dropped_hub} 条路径（中间节点度数 > {_HUB_DEGREE_THRESHOLD}）")

        # 合并两条通道的结果。
        # 搭配路径【不过 hub 剪枝】：它的中间节点按构造就是 Recipe（共现的见证者），
        # 剪掉它等于剪掉这条通道的全部语义；而且 Recipe 度数实测 3~30，远低于阈值 120，
        # 真正需要防的 hub（Category '调料' 1556、'蔬菜' 605）根本不会出现在 食材-菜谱-食材 里。
        # 排序：按 relevance_score 降序，同分时 Python 稳定排序保留输入顺序 ——
        # 这里把搭配路径排在前面，所以同分优先保留搭配证据（它是本问句的直接答案）。
        merged = pairing_paths + pruned
        merged.sort(key=lambda p: p.relevance_score, reverse=True)
        return merged[:_MAX_PATH_RESULTS]

    def _expand_pairing_paths(self, graph_query: GraphQuery, session,
                              seeds: List[Dict[str, Any]],
                              target_keywords: List[str],
                              budget: int) -> List[GraphPath]:
        """「搭配共现」通道：用"经由菜谱共现"的两跳结构找出真正的搭配对象。

        为什么需要单独一条通道：
        变长路径 `(source)-[*1..N]-(target)` 不区分"经由什么连接"，而"X 配什么 Y"问句的语义
        恰恰是【X 和 Y 出现在同一个菜谱里】。用变长路径回答这类问题有两处结构性错配：

        1. 路径预算被种子数瓜分。`_MAX_DEEP_SEED_COUNT` 把参与深遍历的种子收口到 2 个，
           而每个食材种子只能走到"自己所在的那个菜谱"，于是 60 条路径预算全落在 1~2 个菜谱上。
           实测「鸡肉配什么蔬菜好？」只返回 2 条路径，全部来自「麻辣香锅」这一个菜谱 ——
           而图谱里与鸡肉共现的蔬菜有 27 种。
        2. 目标侧必须靠"路径终点自己匹配关键词"来收敛。终点是食材时要靠 category 兜住，
           而"最短路径优先"的截断又会先填满 1 跳路径（食材→菜谱、食材→分类），
           真正有信息量的"食材→菜谱→食材"反而排在后面。

        本通道把这两点一次性解决：
          * 源侧按【名字集合】锚定（用户原词 + 同族别名 + 已解析出的种子名），
            因此同名反规范化副本【全部】参与，不受种子数上限影响；
          * 目标侧在 Cypher 内就约束住（名字/分类/所属 Category 节点），
            不依赖事后剪枝；
          * 聚合与排序交给 Neo4j：按"共现菜谱数"降序，
            即"最常和鸡肉一起出现的蔬菜"排在最前 —— 这正是"配什么"问句想要的答案顺序。

        成本（实测，12 个鸡肉食材名 -> 蔬菜）：6~34ms、返回 27 个搭配；
        同一场景下 3 跳变长路径要枚举 3963 行、680ms。

        Args:
            graph_query: 查询对象（只读 source_entities，不修改）
            session: 已建立的 Neo4j 会话
            seeds: _resolve_seed_nodes 解析出的种子（用于补充源侧名字）
            target_keywords: 已规范化的目标关键词（非空，由调用侧保证）
            budget: 本通道最多产出的搭配路径数

        Returns:
            GraphPath 列表（path_type='multi_hop_pairing'，已按共现强度降序、确定性排序）
        """
        budget = max(1, int(budget))
        raw_names = _dedup_preserve_order([
            self._normalize_entity_name(e) for e in (getattr(graph_query, "source_entities", None) or [])
            if self._normalize_entity_name(e)
        ])
        if not raw_names:
            return []

        # 源侧名字集合 = 用户原词 + 同族别名 + 已解析出的食材种子名
        # （三者取并集而不是只取其一：原词保证"用户说的词"一定在内，
        #   别名保证"图谱里的同族说法"被覆盖，种子名保证"解析结果"不会因为别名被截断而丢失）
        source_names = _dedup_preserve_order(
            raw_names
            + self._expand_alias_names(raw_names)
            + [str(s.get("name") or "") for s in (seeds or [])
               if "Ingredient" in (s.get("labels") or []) and s.get("name")]
        )
        source_names = [n for n in source_names if n]
        if not source_names:
            return []

        # 见证（witness）取多条再在 Python 侧取确定性的一条：
        # Cypher 的 collect 顺序由执行计划决定，不保证稳定；而检索结果必须可复现。
        cypher = """
        MATCH (i:Ingredient)<-[r1:REQUIRES]-(r:Recipe)-[r2:REQUIRES]->(v:Ingredient)
        WHERE i.name IN $source_names
          AND i.nodeId IS NOT NULL AND v.nodeId IS NOT NULL AND r.nodeId IS NOT NULL
          AND NOT coalesce(i.nodeId, '') IN $placeholder_ids
          AND NOT coalesce(v.nodeId, '') IN $placeholder_ids
          AND NOT coalesce(r.nodeId, '') IN $placeholder_ids
          AND i.nodeId <> v.nodeId
          AND any(k IN $target_keywords WHERE
                    toLower(coalesce(v.name, '')) CONTAINS k
                 OR toLower(coalesce(v.category, '')) CONTAINS k
                 // 图谱不变量（实测 2905/2905）：v.category 与它所属 Category 节点的名字完全一致，
                 // 所以上面一条已足够。这里保留类别边兜底，是为了"未来导入的食材缺 category 属性"
                 // 时仍能命中；实测加上它耗时无可见变化（21ms vs 25ms，均在噪声内）。
                 OR EXISTS { MATCH (v)-[:BELONGS_TO_CATEGORY]->(c:Category)
                             WHERE toLower(coalesce(c.name, '')) CONTAINS k })
        // 按【名字】聚合，而不是按节点：食材节点按菜谱反规范化复制，
        // 「姜」在图谱里有 64 个节点，按节点聚合会把 6 个菜谱的共现拆成 6 条 support=1 的记录。
        WITH v.name AS v_name, v.category AS v_cat,
             count(DISTINCT r) AS support,
             count(DISTINCT i.name) AS src_name_count,
             collect(DISTINCT {i: i, r: r, r1: r1, r2: r2, v: v})[0..$witness] AS witnesses
        ORDER BY support DESC, src_name_count DESC, v_name ASC
        LIMIT $limit
        UNWIND witnesses AS w
        RETURN v_name, v_cat, support, src_name_count,
               w.i AS i_node, w.r AS r_node, w.r1 AS r1_rel, w.r2 AS r2_rel,
               // 目标节点【随见证一起取回】而不是用 r2.end_node 现取：
               // 驱动对关系的 end_node 是按需装配的，若该节点未随记录返回，
               // 在结果集消费完后访问可能拿不到（甚至报错），而这里对失败的容错是「静默跳过」，
               // 一旦发生就是整条通道全部丢结果却毫无痕迹。
               w.v AS v_node
        """

        rows: List[Dict[str, Any]] = []
        try:
            rows = [dict(record) for record in session.run(cypher, {
                "source_names": source_names,
                "target_keywords": list(target_keywords),
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "witness": _PAIRING_WITNESS_LIMIT,
                "limit": budget * _PAIRING_ROW_FACTOR,
            })]
        except Exception as e:
            logger.error(f"搭配共现查询失败: {e}")
            return []

        if not rows:
            return []

        # 按 (搭配对象名, 类目) 归并见证行，并选出唯一一条见证三元组（按 nodeId 取最小，保证可复现）
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("v_name") or ""), str(row.get("v_cat") or ""))
            entry = grouped.setdefault(key, {
                "support": int(row.get("support") or 0),
                "src_name_count": int(row.get("src_name_count") or 0),
                "witnesses": [],
            })
            i_node, r_node, v_node = row.get("i_node"), row.get("r_node"), row.get("v_node")
            if i_node is None or r_node is None or v_node is None:
                continue
            entry["witnesses"].append({
                "i_id": str(i_node.get("nodeId") or ""),
                "r_id": str(r_node.get("nodeId") or ""),
                "i_node": i_node, "r_node": r_node, "v_node": v_node,
                "r1": row.get("r1_rel"), "r2": row.get("r2_rel"),
            })

        paths: List[GraphPath] = []
        for (v_name, _v_cat), entry in grouped.items():
            witnesses = [w for w in entry["witnesses"] if w["r1"] is not None and w["r2"] is not None]
            if not witnesses:
                continue
            witness = sorted(witnesses, key=lambda w: (w["r_id"], w["i_id"]))[0]

            # 见证三元组 (源食材, 菜谱, 目标食材) 三个节点都直接来自同一条记录，
            # 因此「描述出来的路径」与「support 统计所依据的那条路径」必然一致，
            # 不会出现"统计来自 A 菜谱、展示却用了 B 菜谱"的错配。
            nodes = [
                self._node_to_dict(witness["i_node"]),
                self._node_to_dict(witness["r_node"]),
                self._node_to_dict(witness["v_node"]),
            ]
            if not all(nodes):
                continue

            rels = [self._rel_to_dict(witness["r1"]), self._rel_to_dict(witness["r2"])]
            path = self._build_graph_path(nodes=nodes, rels=rels, path_type="multi_hop_pairing")
            if not path:
                continue

            # 共现强度写进第二条关系（食材-菜谱-食材 里的最后一条）：
            # 它是"这条路径为什么值得信"的额外证据，_build_path_description 会把它渲染进正文，
            # 让生成模型能说出"在 N 个菜谱里和鸡肉一起出现"这种可核查的表述。
            rels[-1]["pairing_support"] = entry["support"]
            rels[-1]["pairing_source_names"] = entry["src_name_count"]
            paths.append(path)

        # 度数必须在 _score_path 之前补齐：_score_path 的「节点信息量」分量直接读 n["degree"]，
        # 批量补度数一次往返即可（见 _attach_degrees 的性能说明），不要逐节点查。
        self._attach_degrees([n for p in paths for n in p.nodes])

        for path in paths:
            path.relevance_score = round(
                min(self._score_path(path) + self._pairing_support_bonus(path), 0.99), 6
            )

        # 二次排序：Cypher 已按 (support, src_name_count, v_name) 排好，
        # 这里再按同一口径显式排序一次，避免归并（dict 插入序）与 Cypher 顺序不一致时的歧义。
        paths.sort(key=lambda p: (-float(p.relationships[-1].get("pairing_support") or 0),
                                  -float(p.relationships[-1].get("pairing_source_names") or 0),
                                  str(p.nodes[-1].get("name") or "")))
        logger.info(
            f"搭配共现通道：源名字 {len(source_names)} 个 -> 命中 {len(paths)} 个搭配"
            f"（top: {'、'.join(str(p.nodes[-1].get('name')) for p in paths[:5])}）"
        )
        return paths[:budget]

class PathQueryMixin:
    """定向路径查询：实体关系、最短路径与相似实体三条策略的具体实现。"""

    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系（ENTITY_RELATION 分支的真实实现）。

        策略是「先直连、再两跳」：
          1. 先找种子实体与其邻居之间的直连关系；给了目标实体就按名称/分类做约束；
          2. 直连为空且给定目标实体时，放宽到两跳（A -r1- mid -r2- B），
             这对应「A 和 B 之间通过什么关联」这类问法。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=5, session=session)
        if not seeds:
            return []

        seed_ids = [s["id"] for s in seeds]
        targets = self._resolve_seed_nodes(graph_query.target_entities, limit=5, session=session) \
            if graph_query.target_entities else []
        target_ids = [t["id"] for t in targets]
        target_keywords = self._normalize_target_keywords(graph_query.target_entities)

        paths: List[GraphPath] = []

        # 第一级：直连关系
        # 【为什么必须带 ORDER BY】原先只有裸 LIMIT $limit，截断点由存储层决定：
        # 种子度数大（如 调料 1556）时候选行远超 LIMIT，"留下哪 60 条"在不同运行之间
        # 都不稳定 —— 实测同一查询两次运行可能拿到不同的邻居集合，缓存、评测、排障
        # 全都对不上账。排序键选「邻居度数 ASC」有两重作用：
        #   1) 让截断确定可复现（并列再按 nodeId 兜底）；
        #   2) 优先保留【更具体】的邻居。度数低的节点（一个食材、一个步骤）承载的语义
        #      比度数 1556 的分类节点强得多，等于把 LIMIT 花在信息量最高的那一端。
        direct_cypher = """
        MATCH (src) WHERE src.nodeId IN $seed_ids
        MATCH (tgt)-[r]-(src)
        WHERE (tgt:Recipe OR tgt:Ingredient OR tgt:CookingStep OR tgt:Category)
          AND NOT coalesce(tgt.nodeId, '') IN $placeholder_ids
          AND ($target_ids = [] OR tgt.nodeId IN $target_ids)
          AND ($target_keywords = []
               OR any(k IN $target_keywords WHERE
                    toLower(coalesce(tgt.name, '')) CONTAINS k
                    OR toLower(coalesce(tgt.category, '')) CONTAINS k))
        WITH src, tgt, r, COUNT { (tgt)--() } AS tgt_degree
        RETURN src, tgt, r
        ORDER BY tgt_degree ASC, coalesce(tgt.nodeId, ''), type(r)
        LIMIT $limit
        """
        try:
            for record in session.run(direct_cypher, {
                "seed_ids": seed_ids,
                "target_ids": target_ids,
                "target_keywords": target_keywords,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "limit": _MAX_PATH_RESULTS,
            }):
                path = self._build_graph_path(
                    nodes=[self._node_to_dict(record["src"]), self._node_to_dict(record["tgt"])],
                    rels=[self._rel_to_dict(record["r"])],
                    path_type="entity_relation",
                    path_length=1,
                )
                if path:
                    paths.append(path)
        except Exception as e:
            logger.error(f"实体直连关系查询失败: {e}")

        # 第二级：仅当「指定了目标实体但直连为空」时才放宽到两跳，避免无谓的性能开销
        if not paths and target_keywords:
            # 【两跳必须做 hub 剪枝】这是本方法唯一的兜底出口，也是最容易产出"假捷径"的地方：
            # 中间节点只要是 Category 这类超高度数的分类节点，任意两个同分类实体之间都会
            # "接通"。实测「生抽 vs 蚝油」：直连 0 行转入本查询，匹配集共 221 行、
            # 220 行都经过 Category '调料'（唯一个节点、度数 1556），Cypher LIMIT 60
            # 截断后返回的 60 条里有 59 条是
            # 生抽 -[BELONGS_TO_CATEGORY]-> 调料(1556) <-[BELONGS_TO_CATEGORY]- 蚝油。
            # 且 _dedup_paths 一条都合并不掉 —— 它的键是节点序列，而目标侧因为食材
            # 按菜谱反规范化复制而生成了 60 个同名不同 nodeId 的节点，60 条路径的键互不相同。
            # 结果就是 top_k 被同一条语义的重复路径彻底挤占。
            # 剪枝口径与 _expand_multi_hop_paths / _find_shortest_paths 保持一致
            #（中间节点度数 > _HUB_DEGREE_THRESHOLD 即视为 hub，不承担中转语义）。
            # 注意：剪枝的只是"中间节点"，端点不做限制，所以真实的两跳关系不受影响。
            two_hop_cypher = """
            MATCH (src) WHERE src.nodeId IN $seed_ids
            MATCH (src)-[r1]-(mid)-[r2]-(tgt)
            WHERE tgt <> src
              AND (mid:Recipe OR mid:Ingredient OR mid:CookingStep OR mid:Category)
              AND (tgt:Recipe OR tgt:Ingredient OR tgt:CookingStep OR tgt:Category)
              AND NOT coalesce(mid.nodeId, '') IN $placeholder_ids
              AND NOT coalesce(tgt.nodeId, '') IN $placeholder_ids
              AND any(k IN $target_keywords WHERE
                    toLower(coalesce(tgt.name, '')) CONTAINS k
                    OR toLower(coalesce(tgt.category, '')) CONTAINS k)
            WITH src, mid, tgt, r1, r2, COUNT { (mid)--() } AS mid_degree
            WHERE mid_degree <= $hub_degree_threshold
            RETURN src, mid, tgt, r1, r2
            ORDER BY mid_degree ASC, coalesce(tgt.nodeId, ''), coalesce(mid.nodeId, '')
            LIMIT $limit
            """
            try:
                for record in session.run(two_hop_cypher, {
                    "seed_ids": seed_ids,
                    "target_keywords": target_keywords,
                    "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                    "hub_degree_threshold": _HUB_DEGREE_THRESHOLD,
                    "limit": _MAX_PATH_RESULTS,
                }):
                    path = self._build_graph_path(
                        nodes=[self._node_to_dict(record["src"]),
                               self._node_to_dict(record["mid"]),
                               self._node_to_dict(record["tgt"])],
                        rels=[self._rel_to_dict(record["r1"]), self._rel_to_dict(record["r2"])],
                        path_type="entity_relation",
                        path_length=2,
                    )
                    if path:
                        paths.append(path)
            except Exception as e:
                logger.error(f"实体两跳关系查询失败: {e}")

        if not paths:
            return []

        self._attach_degrees([n for p in paths for n in p.nodes])
        for path in paths:
            path.relevance_score = self._score_path(path)
        return paths[:_MAX_PATH_RESULTS]

    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径（PATH_FINDING 分支的真实实现）。

        用 Cypher 原生 shortestPath 求两端点间的最短通路，并强制：
          - 路径上所有节点都必须属于【可作中转的】标签白名单（_PATH_LABEL_FILTER_X，刻意不含
            ConceptType 与 DifficultyLevel）—— 实测不加这层过滤时，最短路径会穿过
            name 为 'Recipe' / 'Ingredient' 的层次节点，得到毫无业务意义的"路径"；
          - 排除占位脏节点（NULL 安全的 coalesce 写法，否则 Category 会被三值逻辑误杀）；
          - 上界同样只能以字面量内插（shortestPath 也不接受参数化上界）；
          - Python 侧再做一次 hub 剪枝（与多跳分支保持一致）：
            实测「宫保鸡丁 → 水煮鱼」的最短路径会返回 `宫保鸡丁 -DIFFICULTY_LEVEL-> 四星 -DIFFICULTY_LEVEL-> 水煮鱼`，
            它唯一的"知识"就是两道菜都是四星。白名单已把 DifficultyLevel 挡在外面，
            这里再加一道度数剪枝，防止修好 NULL 过滤后度数 1556 的 Category 变成新的捷径。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=3, session=session)
        if not seeds:
            return []

        targets = self._resolve_seed_nodes(graph_query.target_entities, limit=3, session=session)
        if not targets:
            logger.info(f"最短路径未解析出目标实体: {graph_query.target_entities}")
            return []

        seed_ids = [s["id"] for s in seeds]
        target_ids = [t["id"] for t in targets]

        # 上界钳制到 [2,6]：太短找不到通路，太长则 shortestPath 的搜索代价陡增
        max_len = _clamp(getattr(graph_query, "max_depth", 4) + 1, 2, 6, 5)

        cypher = f"""
        MATCH (a) WHERE a.nodeId IN $seed_ids
        MATCH (b) WHERE b.nodeId IN $target_ids
        MATCH path = shortestPath((a)-[*..{max_len}]-(b))
        WHERE all(x IN nodes(path) WHERE ({_PATH_LABEL_FILTER_X}))
          AND all(x IN nodes(path) WHERE NOT coalesce(x.nodeId, '') IN $placeholder_ids)
        RETURN nodes(path) AS pnodes, relationships(path) AS prels, length(path) AS path_len
        LIMIT $limit
        """

        paths: List[GraphPath] = []
        try:
            for record in session.run(cypher, {
                "seed_ids": seed_ids,
                "target_ids": target_ids,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "limit": _MAX_PATH_RESULTS,
            }):
                path = self._build_graph_path(
                    nodes=[self._node_to_dict(n) for n in (record.get("pnodes") or [])],
                    rels=[self._rel_to_dict(r) for r in (record.get("prels") or [])],
                    path_type="shortest_path",
                    path_length=record.get("path_len"),
                )
                if path:
                    paths.append(path)
        except Exception as e:
            logger.error(f"最短路径查询失败: {e}")
            return []

        if not paths:
            return []

        self._attach_degrees([n for p in paths for n in p.nodes])

        # hub 剪枝：丢弃中间节点度数超阈值的路径（与 _expand_multi_hop_paths 口径一致）
        pruned: List[GraphPath] = []
        for path in paths:
            if any(self._is_hub(n) for n in path.nodes[1:-1]):
                logger.info("最短路径经过 hub 中间节点，已丢弃："
                            + " → ".join(str(n.get("name")) for n in path.nodes))
                continue
            path.relevance_score = self._score_path(path)
            pruned.append(path)
        return self._dedup_paths(pruned)

    def _find_similar_entities(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找相似实体（CLUSTERING 分支的真实实现）。

        图谱实测结论决定了这里的三条通路（注意：**不能**依赖共享食材，因为食材节点是按菜谱私有的）：
          1. SIMILAR 显式相似边（精确但稀疏，全库仅约 46 道菜带此边）；
          2. 同 Category 分类（最主要的跨菜谱通路，如「水产」下有 23 道菜）；
          3. 同 cuisineType 菜系（属性级同组，如「川菜」32 道）。
        三者按 1 -> 2 -> 3 的优先级取，前一条通路够用就不再往下取，避免结果被大面积泛化的菜系刷屏。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=3, session=session)
        if not seeds:
            return []

        seed_ids = [s["id"] for s in seeds]
        limit = _clamp(getattr(graph_query, "max_nodes", 30), 5, _MAX_SUBGRAPH_NODES, 30)

        queries = [
            # 通路 1：显式 SIMILAR 边
            ("""
            UNWIND $seed_ids AS sid
            MATCH (src:Recipe) WHERE src.nodeId = sid
            MATCH (src)-[r:SIMILAR]-(peer:Recipe)
            WHERE peer.nodeId <> src.nodeId
            RETURN src, peer, r, [] AS mid_nodes, [] AS mid_rels
            LIMIT $limit
            """, "cluster_similar"),
            # 通路 2：同 Category 分类
            ("""
            UNWIND $seed_ids AS sid
            MATCH (src:Recipe) WHERE src.nodeId = sid
            MATCH (src)-[r1:BELONGS_TO_CATEGORY]->(c:Category)<-[r2:BELONGS_TO_CATEGORY]-(peer:Recipe)
            WHERE peer.nodeId <> src.nodeId
            RETURN src, peer, null AS r, [c] AS mid_nodes, [r1, r2] AS mid_rels
            LIMIT $limit
            """, "cluster_category"),
            # 通路 3：同 cuisineType 菜系（属性级，无中间节点）
            ("""
            UNWIND $seed_ids AS sid
            MATCH (src:Recipe) WHERE src.nodeId = sid
            MATCH (peer:Recipe)
            WHERE peer.cuisineType = src.cuisineType
              AND peer.cuisineType IS NOT NULL
              AND peer.nodeId <> src.nodeId
            RETURN src, peer, null AS r, [] AS mid_nodes, [] AS mid_rels
            LIMIT $limit
            """, "cluster_cuisine"),
        ]

        for cypher, path_type in queries:
            paths: List[GraphPath] = []
            try:
                for record in session.run(cypher, {"seed_ids": seed_ids, "limit": limit}):
                    mid_nodes = [self._node_to_dict(n) for n in (record.get("mid_nodes") or [])]
                    mid_rels = [self._rel_to_dict(r) for r in (record.get("mid_rels") or [])]
                    rels = list(mid_rels)
                    if record.get("r") is not None:
                        rels.append(self._rel_to_dict(record["r"]))

                    path = self._build_graph_path(
                        nodes=[self._node_to_dict(record["src"])] + mid_nodes + [self._node_to_dict(record["peer"])],
                        rels=rels,
                        path_type=path_type,
                    )
                    if path:
                        paths.append(path)
            except Exception as e:
                logger.error(f"相似实体查询失败（{path_type}）: {e}")
                continue

            if paths:
                self._attach_degrees([n for p in paths for n in p.nodes])
                for path in paths:
                    path.relevance_score = self._score_path(path)
                logger.info(f"聚类命中通路 {path_type}，返回 {len(paths)} 条")
                return paths[:_MAX_PATH_RESULTS]

        return []
