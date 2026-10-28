"""子图抽取与推理：两阶段知识子图抽取与降级、基于拓扑模式的推理链构造与校验。"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .shared_constants_helpers_models import (
    GraphQuery,
    KnowledgeSubgraph,
    _MAX_SUBGRAPH_NODES,
    _MAX_TRAVERSAL_DEPTH,
    _MIN_SUBGRAPH_NODES,
    _PLACEHOLDER_NAMES,
    _PLACEHOLDER_NODE_IDS,
    _SUBGRAPH_CACHE_LIMIT,
    _char_bigrams,
    _clamp,
    _containment,
    _dedup_preserve_order,
)

logger = logging.getLogger(__name__)


class SubgraphExtractionMixin:
    """知识子图：两阶段邻域抽取、行组装、缓存淘汰与三级降级。"""

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """
        提取知识子图：获取实体相关的完整知识网络
        这体现了图RAG的整体性思维

        返回子图的完整性由 attached 的 graph_metrics 如实标注，调用方应以指标为准，
        不要假设"传了 max_depth 就一定能取到那么深"：
          - depth_reached：实际达到的跳数。抽取采用【两阶段】设计（种子 1 跳邻域 +
            从非 hub 邻居再扩 1 跳），因此 depth_reached 的实测上限是 2；
            即便 max_depth 传 3，也只是沿用同一套两阶段抽取（第 3 跳会让子图被 hub
            迅速拉成全图，收益远低于噪声），此时 depth_reached 仍会诚实回报 2。
          - hub_pruned：在【阶段二前沿】上因度数超阈值被剪掉的节点数。
            它只统计"本来可以继续外扩、但被判定为 hub 而不再展开"的邻居，
            因此二跳新进来的高度数节点【不计入】该值（它们不再向外扩展，无需剪枝）。
            换句话说 hub_pruned=0 不代表子图里没有高度数节点，
            只代表一跳邻域里没有需要拦截的外扩入口。

        异常处理：主路径（含两阶段查询）整体失败时，不会返回空壳，而是转入
        _fallback_subgraph_extraction 做三级真实降级，并在 metrics 里标注 degraded 来源。
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")

        if not self._is_driver_ready():
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)

        # 缓存命中：子图抽取较重（含多轮邻域扩展），同一实体组合重复提问时可直接复用
        cache_key = self._subgraph_cache_key(graph_query)
        cached = self.subgraph_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"子图命中缓存: {cache_key[:60]}")
            return cached

        depth = _clamp(getattr(graph_query, "max_depth", 2), 1, _MAX_TRAVERSAL_DEPTH, 2)
        max_nodes = _clamp(getattr(graph_query, "max_nodes", 50), _MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES, 50)

        subgraph: Optional[KnowledgeSubgraph] = None
        try:
            with self.driver.session() as session:
                subgraph = self._collect_subgraph(session, graph_query, depth, max_nodes)
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
            subgraph = None

        # 主路径失败则走真实降级（而不是返回空壳），保证调用方拿到的至少是可用证据
        if subgraph is None or not subgraph.central_nodes:
            logger.warning("子图主路径未取得结果，执行降级抽取")
            subgraph = self._fallback_subgraph_extraction(graph_query)

        # 【降级结果不入缓存】
        # 缓存键只含 query_type / 源实体 / 目标实体 / 深度 / 节点预算，不含"本次是否降级"，
        # 而命中缓存的路径既不重新抽取也不刷新 FIFO 位置。若把降级子图写进去，
        # 一次瞬时故障（例如 Neo4j 重启期间 driver 非 None 但查询全失败，
        # _collect_subgraph 内部吞掉异常返回 None）就会把 degraded=1.0 的空壳
        # 长期钉在缓存里：Neo4j 恢复后同一 (实体, 深度, max_nodes) 组合仍然命中该壳子，
        # 表现为"图库明明好了，这个实体却一直检索不到"。
        # 代价是降级期间同一查询会重复走降级路径 —— 降级本就应当尽快被上层观测到，
        # 这个代价是刻意接受的。
        if subgraph is not None and float(subgraph.graph_metrics.get("degraded") or 0.0) > 0.0:
            logger.warning(f"子图为降级结果（degraded={subgraph.graph_metrics.get('degraded')}），不写入缓存: {cache_key[:60]}")
        else:
            self._cache_subgraph(cache_key, subgraph)
        return subgraph

    def _subgraph_cache_key(self, graph_query: GraphQuery) -> str:
        """构造子图缓存键：涵盖类型、实体、深度与节点预算，任一变化都必须重新抽取。"""
        return "|".join([
            str(getattr(graph_query.query_type, "value", graph_query.query_type)),
            ",".join(sorted(graph_query.source_entities or [])),
            ",".join(sorted(graph_query.target_entities or [])),
            str(_clamp(getattr(graph_query, "max_depth", 2), 1, _MAX_TRAVERSAL_DEPTH, 2)),
            str(_clamp(getattr(graph_query, "max_nodes", 50), _MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES, 50)),
        ])

    def _cache_subgraph(self, cache_key: str, subgraph: KnowledgeSubgraph) -> None:
        """写入子图缓存并做 FIFO 淘汰，防止长时间运行内存无界增长。"""
        if subgraph is None:
            return
        if cache_key not in self.subgraph_cache:
            self._subgraph_cache_order.append(cache_key)
        self.subgraph_cache[cache_key] = subgraph
        while len(self._subgraph_cache_order) > _SUBGRAPH_CACHE_LIMIT:
            oldest = self._subgraph_cache_order.popleft()
            self.subgraph_cache.pop(oldest, None)

    def _collect_subgraph(self, session, graph_query: GraphQuery, depth: int,
                          max_nodes: int) -> Optional[KnowledgeSubgraph]:
        """两阶段抽取知识子图（内部实现，替代原先有语法/语义缺陷的单条聚合 Cypher）。

        原实现的三个问题在这里被绕开：
          - `MATCH (source)-[r*1..n]-(neighbor)` 之后 `collect(DISTINCT r)` 收集的是「关系列表的列表」，
            后续 dict(rel) 必然报错 —— 本实现改为【逐行返回 (种子, 关系, 邻居) 三元组】，在 Python 侧组装；
          - 在 collect 之后再用 WHERE 过滤聚合结果 —— 语义不成立，本实现改为在 RETURN 阶段直接 LIMIT；
          - 只调用 result.single()，多源实体时会丢结果 —— 本实现按行累积，天然支持多实体。

        阶段一：解析种子并取其 1 跳邻域（含边属性与方向）；
        阶段二：若 depth >= 2，再从度数较低的邻居（非 hub）向外扩一跳，把"知识的第二圈"补进来。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=5, session=session)
        if not seeds:
            return None

        seed_ids = [s["id"] for s in seeds]
        # 行数上限：每个种子平均取若干条边即可覆盖其邻域，避免种子度数过大时把结果撑爆
        row_limit = max(max_nodes * 4, 50)

        # ---------------- 阶段一：种子 + 1 跳邻域 ----------------
        hop1_cypher = """
        MATCH (seed) WHERE seed.nodeId IN $seed_ids
        MATCH (seed)-[r]-(nb)
        WHERE (nb:Recipe OR nb:Ingredient OR nb:CookingStep OR nb:Category)
          AND NOT coalesce(nb.nodeId, '') IN $placeholder_ids
        RETURN
            seed.nodeId AS seed_id, seed.name AS seed_name, labels(seed) AS seed_labels,
            properties(seed) AS seed_props, COUNT { (seed)--() } AS seed_degree,
            type(r) AS rel_type, properties(r) AS rel_props,
            startNode(r).nodeId AS rel_start, startNode(r).name AS rel_start_name,
            endNode(r).nodeId AS rel_end, endNode(r).name AS rel_end_name,
            nb.nodeId AS nb_id, nb.name AS nb_name, labels(nb) AS nb_labels,
            properties(nb) AS nb_props, COUNT { (nb)--() } AS nb_degree
        ORDER BY nb_degree DESC
        LIMIT $row_limit
        """

        hop1_rows: List[Dict[str, Any]] = []
        preloaded_seed_rows: List[Dict[str, Any]] = []

        try:
            hop1_rows = [dict(record) for record in session.run(hop1_cypher, {
                "seed_ids": seed_ids,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "row_limit": row_limit,
            })]
        except Exception as e:
            logger.error(f"子图阶段一查询失败: {e}")
            return None

        # 【多种子的行序轮转】hop1_rows 按 nb_degree DESC 全局排序，而 _assemble_from_rows
        # 是按行序贪心填充 connected_nodes（填满 max_nodes 即停），于是种子之间的度数差异
        # 会直接变成预算分配差异：实测 ['宫保鸡丁','水煮鱼'] max_nodes=50 时，
        # 水煮鱼（一跳邻居 42 个、度数普遍 58~83）把预算吃到 40，宫保鸡丁只拿到 10，
        # 多实体问题被单实体答案淹没。这里只把行序按种子轮转（每个种子内部仍保持度数降序），
        # 让贪心填充近似变成"按种子均分预算"，且每个种子的高价值邻居仍优先入选。
        # 保留全局排序不变的部分：单种子（绝大多数查询）时轮转是恒等操作，行为逐字节不变。
        # 局限：Cypher 侧的 LIMIT $row_limit 仍是全局截断，种子度数极端悬殊时截断本身
        # 就可能把某个种子整片切掉；根治需要按种子分别下发查询，代价是查询数随种子数增长。
        hop1_rows = self._interleave_rows_by_seed(hop1_rows)

        central_nodes, connected_nodes, relationships = self._assemble_from_rows(hop1_rows, max_nodes)

        # 种子若因度数排序被挤掉，单独补一次轻量查询，保证 central_nodes 一定有内容
        # 用身份键比对：种子本身都来自 entity_cache（必有 nodeId），此处只是保持与
        # 子图内部其他集合运算一致的标识口径，避免出现两套 id 语义
        existing_ids = {self._node_identity_key(n) for n in central_nodes}
        missing_seed_ids = [sid for sid in seed_ids if sid not in existing_ids]
        if missing_seed_ids:
            try:
                preloaded_seed_rows = [dict(record) for record in session.run(
                    """
                    MATCH (n) WHERE n.nodeId IN $ids
                    RETURN n.nodeId AS seed_id, n.name AS seed_name, labels(n) AS seed_labels,
                           properties(n) AS seed_props, COUNT { (n)--() } AS seed_degree
                    """,
                    {"ids": missing_seed_ids},
                )]
            except Exception as e:
                logger.warning(f"补查种子节点失败: {e}")

            for row in preloaded_seed_rows:
                node = {
                    "id": str(row.get("seed_id") or ""),
                    "node_id": str(row.get("seed_id") or ""),
                    "name": str(row.get("seed_name") or ""),
                    "labels": list(row.get("seed_labels") or []),
                    "label": (list(row.get("seed_labels") or []) or [""])[0],
                    "properties": dict(row.get("seed_props") or {}),
                    "degree": int(row.get("seed_degree") or 0),
                }
                if node["id"] and self._is_meaningful_node(node):
                    central_nodes.append(node)

        # 【中心节点 / 邻居节点的跨列表收口】
        # central_nodes 与 connected_nodes 各自用一套去重集合（seen_central / seen_connected），
        # 两套互不相通，因此当一个被解析出来的种子同时又是另一个种子的邻居时，
        # 同一个节点会同时留在两个列表里。后果是 node_count = len(central)+len(connected)
        # 重复计数，进而把 density = 2E/(N(N-1)) 压低、把 avg_degree 抬高，
        # 并把这个虚高的节点数通过 _subgraph_struct_score 的 coverage 项影响排序、
        # 写进喂给 LLM 的子图描述文本。
        # 触发条件不是"传了 2~3 个实体"，而是"解析出的种子集合内部存在直接相邻的一对"：
        # 单个实体也能触发 —— _resolve_seed_nodes(['红烧鱼']) 会同时返回 红烧鱼 与 红烧鱼头，
        # 二者之间有 SIMILAR 边。
        # 收口口径：已作为中心节点的节点不再计入邻居列表（中心节点的语义优先级更高），
        # 但它的关系【全部保留】，只去掉重复计数。
        central_keys = {self._node_identity_key(n) for n in central_nodes}
        central_keys.discard("")
        if central_keys:
            before = len(connected_nodes)
            connected_nodes = [n for n in connected_nodes
                               if self._node_identity_key(n) not in central_keys]
            if len(connected_nodes) != before:
                logger.info(f"中心节点与邻居节点重叠收口：{before - len(connected_nodes)} 个种子节点不再重复计入邻居")

        # ---------------- 阶段二：从非 hub 邻居再扩一跳 ----------------
        depth_reached = 1
        hub_pruned = 0
        if depth >= 2:
            # current_ids 用「身份键」而非裸 nodeId：无 nodeId 的节点（Category 等）用裸 nodeId
            # 会全部塌缩成空串，导致阶段二的 exclude 把不同节点误判成同一个，
            # 已有的 Category 邻居也无法被正确识别为"已收录"。
            current_ids = {self._node_identity_key(n) for n in central_nodes + connected_nodes}
            current_ids.discard("")
            # 只从度数较低的邻居继续扩展：hub 邻居一旦展开会把整个图谱拉进来
            frontier = [n for n in connected_nodes if not self._is_hub(n)]
            hub_pruned = len(connected_nodes) - len(frontier)
            frontier = frontier[:max(1, max_nodes // 2)]
            # 前沿节点分两路下发：有 nodeId 的走 nodeId；无 nodeId 的（Category / ConceptType 等，
            # 实测 nodeId 全为 NULL）只能按 name 匹配。早期实现只传 nodeId，
            # 这类节点在阶段二【永远无法被再次展开】，荤菜 这种"近 hub"主题节点
            # 下面的第二圈知识会整片静默丢失。
            frontier_ids = [str(n.get("id") or "") for n in frontier if n.get("id")]
            frontier_names = [str(n.get("name") or "") for n in frontier if not n.get("id")]
            frontier_names = [nm for nm in frontier_names if nm]

            if frontier_ids or frontier_names:
                # 前沿条件按需拼装：只有当确实存在"无 nodeId 的前沿节点"时才引入 OR。
                # 实测本图上 OR 分支会让查询计划无法纯走 nodeId 索引
                # （同等工作量 59ms -> 96ms），而绝大多数种子（Recipe）的邻居全是
                # Ingredient / CookingStep，都带 nodeId，走不到 OR 那一支。
                # `f.nodeId IS NULL` 这个限定不能省：Category 与 RecipeCategory 存在大量同名节点
                # （水产 / 荤菜 / 调料 …），少了它就会把带 nodeId 的 RecipeCategory 也一并命中。
                if frontier_names:
                    frontier_where = ("f.nodeId IN $frontier_ids\n"
                                      "                   OR (f.nodeId IS NULL AND f.name IN $frontier_names)")
                else:
                    frontier_where = "f.nodeId IN $frontier_ids"
                hop2_cypher = f"""
                MATCH (f)
                WHERE {frontier_where}
                MATCH (f)-[r]-(nb)
                WHERE (nb:Recipe OR nb:Ingredient OR nb:CookingStep OR nb:Category)
                  AND NOT coalesce(nb.nodeId, '') IN $placeholder_ids
                RETURN
                    f.nodeId AS seed_id, f.name AS seed_name, labels(f) AS seed_labels,
                    properties(f) AS seed_props, COUNT {{ (f)--() }} AS seed_degree,
                    type(r) AS rel_type, properties(r) AS rel_props,
                    startNode(r).nodeId AS rel_start, startNode(r).name AS rel_start_name,
                    endNode(r).nodeId AS rel_end, endNode(r).name AS rel_end_name,
                    nb.nodeId AS nb_id, nb.name AS nb_name, labels(nb) AS nb_labels,
                    properties(nb) AS nb_props, COUNT {{ (nb)--() }} AS nb_degree
                ORDER BY nb_degree ASC
                LIMIT $row_limit
                """
                try:
                    remaining = max(max_nodes - len(current_ids), 0)
                    if remaining > 0:
                        hop2_rows = [dict(record) for record in session.run(hop2_cypher, {
                            "frontier_ids": frontier_ids,
                            "frontier_names": frontier_names,
                            "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                            "row_limit": max(remaining * 2, 20),
                        })]
                        # 阶段二只取新增的邻居与关系；返回的"中心节点"其实就是 frontier 自身，
                        # 它们已经作为一跳邻居存在于 connected_nodes 中，故此处有意忽略（下划线前缀）
                        _extra_central, extra_nodes, extra_rels = self._assemble_from_rows(
                            hop2_rows, remaining, exclude_ids=current_ids,
                        )
                        connected_nodes.extend(extra_nodes)
                        relationships.extend(extra_rels)
                        if extra_nodes:
                            depth_reached = 2
                except Exception as e:
                    logger.warning(f"子图阶段二查询失败（保持一跳结果）: {e}")

        # 【跨阶段关系去重】两个阶段各自的 seen_rel 是调用内局部集合，互相看不见，
        # 二跳回连到种子的边会被重复追加一次。此处按与写入时完全一致的签名口径统一收口，
        # 保证 relationship_count / density 与图结构自洽（详见 _dedupe_relationships）。
        relationships = self._dedupe_relationships(relationships)

        # ---------------- 图指标计算 ----------------
        node_count = len(central_nodes) + len(connected_nodes)
        rel_count = len(relationships)
        # 无向图密度：D = 2E / (N(N-1))，N<=1 时无意义，显式置 0
        density = (2.0 * rel_count) / (node_count * (node_count - 1)) if node_count > 1 else 0.0
        all_degrees = [int(n.get("degree") or 0) for n in central_nodes + connected_nodes]
        avg_degree = sum(all_degrees) / float(len(all_degrees)) if all_degrees else 0.0

        metrics: Dict[str, Any] = {
            "node_count": float(node_count),
            "relationship_count": float(rel_count),
            "density": round(float(min(density, 1.0)), 6),
            "avg_degree": round(avg_degree, 3),
            "max_degree": float(max(all_degrees) if all_degrees else 0),
            "depth_reached": float(depth_reached),
            "hub_pruned": float(hub_pruned),
        }

        return KnowledgeSubgraph(
            central_nodes=central_nodes,
            connected_nodes=connected_nodes,
            relationships=relationships,
            graph_metrics=metrics,
            reasoning_chains=[],
        )

    def _interleave_rows_by_seed(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把「(种子, 关系, 邻居)」行按种子轮转排序，用于多种子场景下的预算公平分配。

        规则：按 seed_id 首次出现的顺序把行分组（组内保持原有顺序，即 nb_degree DESC），
        然后依次从每组取一行轮转输出。单种子（或 rows 少于两个不同种子）时返回原列表，
        保证最常见的查询路径行为完全不变。

        Args:
            rows: 查询返回的扁平三元组（需含 seed_id 列）

        Returns:
            轮转后的新列表（不修改入参）
        """
        if not rows:
            return rows
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("seed_id") or "")
            groups.setdefault(key, []).append(row)
        if len(groups) <= 1:
            return rows
        interleaved: List[Dict[str, Any]] = []
        cursors = {key: 0 for key in groups}
        remaining = len(rows)
        while remaining > 0:
            for key, group in groups.items():
                cursor = cursors[key]
                if cursor < len(group):
                    interleaved.append(group[cursor])
                    cursors[key] = cursor + 1
                    remaining -= 1
        return interleaved

    def _assemble_from_rows(self, rows: List[Dict[str, Any]], max_nodes: int,
                            exclude_ids: Optional[Set[str]] = None) -> Tuple[
                                List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """把「(种子, 关系, 邻居)」三元组行组装成 中心节点 / 邻居节点 / 关系 三段结构。

        Args:
            rows: 查询返回的扁平三元组
            max_nodes: 邻居节点数量上限
            exclude_ids: 需要排除的 nodeId（用于阶段二避免重复纳入已有节点）

        Returns:
            (central_nodes, connected_nodes, relationships)；三段均已完成去重与脏数据过滤
        """
        exclude = exclude_ids or set()
        central_nodes: List[Dict[str, Any]] = []
        connected_nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        seen_central: Set[str] = set()
        seen_connected: Set[str] = set()
        seen_rel: Set[Tuple[str, str, str]] = set()

        for row in rows:
            seed_id = str(row.get("seed_id") or "")
            seed_node = {
                "id": seed_id,
                "node_id": seed_id,
                "name": str(row.get("seed_name") or ""),
                "labels": list(row.get("seed_labels") or []),
                "label": (list(row.get("seed_labels") or []) or [""])[0],
                "properties": dict(row.get("seed_props") or {}),
                "degree": int(row.get("seed_degree") or 0),
            }
            # 种子同样按"身份键"去重：无 nodeId 的节点（Category 等）靠 name 区分，
            # 避免多个不同 Category 都被当成同一个空串节点
            seed_key = self._node_identity_key(seed_node)
            if seed_key and seed_key not in seen_central and self._is_meaningful_node(seed_node):
                central_nodes.append(seed_node)
                seen_central.add(seed_key)

            nb_node = {
                "id": str(row.get("nb_id") or ""),
                "node_id": str(row.get("nb_id") or ""),
                "name": str(row.get("nb_name") or ""),
                "labels": list(row.get("nb_labels") or []),
                "label": (list(row.get("nb_labels") or []) or [""])[0],
                "properties": dict(row.get("nb_props") or {}),
                "degree": int(row.get("nb_degree") or 0),
            }
            # 【关键】这里必须用身份键而不是 nb_id 做真值判断与去重。
            # 原写法 `if not nb_id ... continue` 会把所有无 nodeId 的邻居（Category /
            # ConceptType / DifficultyLevel / RecipeCategory 全部无 nodeId）静默丢弃，
            # 是"子图里 Category 恒为 0"的根因 —— 上游 Cypher 的 NULL 安全过滤已经把
            # 荤菜 这类节点放行了，却在 Python 组装这一步又被判掉。
            nb_key = self._node_identity_key(nb_node)
            if not nb_key:
                # 既无 nodeId 也无 name 的脏节点，无法标识，直接丢弃
                continue
            if nb_key in seen_connected or nb_key in exclude:
                # 节点已收录，本轮不重复加入节点列表；但这一行代表的边仍是新边
                # （平行边 / 跨种子边 / 二跳回连边），必须记录，否则图结构会缺边。
                # 【此处不能再用 len(connected_nodes) < max_nodes 设闸】
                # 进入本分支的前提就是 nb 已在本轮子图内（在 seen_connected 或
                # 在 exclude 指定的已收录集合里），所以补进去的边【两端必然都在子图内】，
                # 不存在"悬空边"风险。此前把补边绑在节点预算上，导致预算一旦用尽，
                # 连子图内部真实存在的关系也被静默丢弃（该分支当时连日志都没有）：
                # 实测单实体「水煮鱼」depth=1 max_nodes=10 时，20 条两端均在子图内的边
                # 只记录了 19 条；多实体场景丢掉的则是独有证据
                #（如 ['鸡蛋','咖喱炒蟹'] 丢 REQUIRES 咖喱炒蟹→鸡蛋）。
                # 节点预算与关系记录是两件事，关系不受节点预算约束。
                self._append_relationship(relationships, seen_rel, row)
                continue
            if len(connected_nodes) >= max_nodes:
                continue
            if not self._is_meaningful_node(nb_node):
                continue
            connected_nodes.append(nb_node)
            seen_connected.add(nb_key)
            self._append_relationship(relationships, seen_rel, row)

        return central_nodes, connected_nodes, relationships

    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象。

        兼容两种输入：
          - 本模块内部使用的规范化载荷（含 central_nodes / connected_nodes / relationships / metrics 四个键）；
          - 早期版本的 neo4j Record（含 source / nodes / rels / metrics 键），保留以兼容既有调用。
        """
        try:
            if isinstance(record, dict) and "central_nodes" in record:
                return KnowledgeSubgraph(
                    central_nodes=list(record.get("central_nodes") or []),
                    connected_nodes=list(record.get("connected_nodes") or []),
                    relationships=list(record.get("relationships") or []),
                    graph_metrics=dict(record.get("metrics") or {}),
                    reasoning_chains=list(record.get("reasoning_chains") or []),
                )

            # 兼容路径：早期 Record 形态
            central_nodes = [self._node_to_dict(record["source"])] if record.get("source") is not None else []
            connected_nodes = [self._node_to_dict(node) for node in (record.get("nodes") or [])]
            relationships = [
                self._rel_to_dict(rel) if not isinstance(rel, list) else self._rel_to_dict(rel[0])
                for rel in (record.get("rels") or [])
                if rel is not None
            ]
            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=dict(record.get("metrics") or {}),
                reasoning_chains=[]
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )

    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取（真实可用的降级，而非空壳）。

        三级降级，逐级放宽：
          1. 图谱侧只做最简单的一次 1 跳邻居查询（不带 COUNT{}、不带聚合、不带路径模式，最大化成功率）；
          2. 失败则用预热的 entity_cache 在内存里构造一个最小子图（Neo4j 抖动时仍能给出实体级证据）；
          3. 全部失败才返回空子图，并在 metrics 里标注降级来源，便于上层观测与排障。
        """
        degraded_metrics: Dict[str, Any] = {"degraded": 1.0}

        # 降级路径的节点预算【必须沿用调用方传入的 max_nodes】。
        # 此前一级降级把上限硬编码为 80 并把 Cypher 的 row_limit 硬编码为 200，
        # 完全忽略 graph_query.max_nodes —— 而调用方（含 graph_rag_search 的透传）
        # 正是靠这个参数控制 prompt 长度。调用方把 max_nodes 压到 10 以适配上下文时，
        # 降级路径反而给出 80 个节点，降级结果比正常结果还重，与"降级应当更轻"的直觉相悖。
        fallback_max_nodes = _clamp(getattr(graph_query, "max_nodes", 50),
                                   _MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES, 50)

        # 一级降级：最简 Cypher
        try:
            with self.driver.session() as session:
                seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=3, session=session)
                if seeds:
                    rows = [dict(r) for r in session.run(
                        """
                        MATCH (seed) WHERE seed.nodeId IN $seed_ids
                        MATCH (seed)-[r]-(nb)
                        WHERE (nb:Recipe OR nb:Ingredient OR nb:CookingStep OR nb:Category)
                          AND NOT coalesce(nb.nodeId, '') IN $placeholder_ids
                        RETURN seed.nodeId AS seed_id, seed.name AS seed_name, labels(seed) AS seed_labels,
                               properties(seed) AS seed_props, 0 AS seed_degree,
                               type(r) AS rel_type, properties(r) AS rel_props,
                               startNode(r).nodeId AS rel_start, startNode(r).name AS rel_start_name,
                               endNode(r).nodeId AS rel_end, endNode(r).name AS rel_end_name,
                               nb.nodeId AS nb_id, nb.name AS nb_name, labels(nb) AS nb_labels,
                               properties(nb) AS nb_props, 0 AS nb_degree
                        LIMIT $row_limit
                        """,
                        {
                            "seed_ids": [s["id"] for s in seeds],
                            "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                            "row_limit": max(fallback_max_nodes * 4, 50),
                        },
                    )]
                    central_nodes, connected_nodes, relationships = self._assemble_from_rows(
                        rows, fallback_max_nodes)
                    # 与主路径同口径的跨列表收口：种子若同时是另一个种子的邻居，
                    # 不能既算中心节点又算邻居，否则 node_count 重复计数
                    _central_keys = {self._node_identity_key(n) for n in central_nodes}
                    _central_keys.discard("")
                    if _central_keys:
                        connected_nodes = [n for n in connected_nodes
                                           if self._node_identity_key(n) not in _central_keys]
                    if central_nodes or connected_nodes:
                        node_count = len(central_nodes) + len(connected_nodes)
                        rel_count = len(relationships)
                        density = (2.0 * rel_count) / (node_count * (node_count - 1)) if node_count > 1 else 0.0
                        self._attach_degrees(central_nodes + connected_nodes)
                        # avg_degree / max_degree 与主路径保持同一套字段：
                        # 描述文本 (_build_subgraph_description) 直接读 metrics['avg_degree']，
                        # 一级降级不写这两个字段时它会走 `metrics.get('avg_degree', 0)` 兜底，
                        # 于是在度数明明已经补好的情况下，喂给生成模型的文本写着「平均度数 0」——
                        # 降级路径本来就最需要自证可信，这种"数据齐全却报 0"的表述反而最伤可信度。
                        _degrees = [int(n.get("degree") or 0) for n in (central_nodes + connected_nodes)]
                        metrics = {
                            "node_count": float(node_count),
                            "relationship_count": float(rel_count),
                            "density": round(float(min(density, 1.0)), 6),
                            "avg_degree": round(sum(_degrees) / float(len(_degrees)), 3) if _degrees else 0.0,
                            "max_degree": float(max(_degrees)) if _degrees else 0.0,
                            "depth_reached": 1.0,
                            "source": 1.0,
                        }
                        metrics.update(degraded_metrics)
                        logger.info(f"子图一级降级成功：{node_count} 节点 / {rel_count} 关系")
                        return KnowledgeSubgraph(
                            central_nodes=central_nodes,
                            connected_nodes=connected_nodes,
                            relationships=relationships,
                            graph_metrics=metrics,
                            reasoning_chains=[],
                        )
        except Exception as e:
            logger.warning(f"子图一级降级失败: {e}")

        # 二级降级：用内存索引构造最小子图
        try:
            names = [self._normalize_entity_name(e) for e in (graph_query.source_entities or [])]
            names = [n for n in names if n]
            cache_seeds = self._resolve_seed_nodes_from_cache([n.lower() for n in names], limit=3) if names else []
            if cache_seeds:
                # 把内存索引里与种子同类/同名的节点也纳入，形成最小可用知识网络
                # 节点预算同样沿用调用方传入的 max_nodes（此前硬编码 20，会无视调用方
                # 为控制 prompt 长度而下调的上限），同时不超过 20 以免降级结果过重
                _level2_budget = min(20, fallback_max_nodes)
                _seed_ids = {str(s.get("id") or "") for s in cache_seeds}
                connected: List[Dict[str, Any]] = []
                for node_id, info in self.entity_cache.items():
                    if len(connected) >= _level2_budget:
                        break
                    node_name = str(info.get("name") or "")
                    if not node_name or node_name in _PLACEHOLDER_NAMES:
                        continue
                    # 种子自身也会命中下面的子串条件（s["name"] in node_name 对 s["name"]==node_name
                    # 恒为真），若不排除就会既算中心节点又算关联节点，node_count 重复计数
                    if str(node_id) in _seed_ids:
                        continue
                    if any(s["name"] and s["name"] in node_name for s in cache_seeds):
                        connected.append({
                            "id": str(node_id),
                            "node_id": str(node_id),
                            "name": node_name,
                            "labels": list(info.get("labels") or []),
                            "label": (list(info.get("labels") or []) or [""])[0],
                            "properties": {"name": node_name, "category": info.get("category")},
                            "degree": int(info.get("degree") or 0),
                        })
                metrics = {
                    "node_count": float(len(cache_seeds) + len(connected)),
                    "relationship_count": 0.0,
                    "density": 0.0,
                    "depth_reached": 0.0,
                    "source": 2.0,
                }
                metrics.update(degraded_metrics)
                logger.info(f"子图二级降级（内存索引）：{len(cache_seeds)} 中心 / {len(connected)} 关联")
                return KnowledgeSubgraph(
                    central_nodes=cache_seeds,
                    connected_nodes=connected,
                    relationships=[],
                    graph_metrics=metrics,
                    reasoning_chains=[],
                )
        except Exception as e:
            logger.warning(f"子图二级降级失败: {e}")

        # 三级降级：确实拿不到任何证据
        logger.error("子图降级全部失败，返回空子图")
        metrics = {"node_count": 0.0, "relationship_count": 0.0, "density": 0.0, "source": 0.0}
        metrics.update(degraded_metrics)
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics=metrics,
            reasoning_chains=[]
        )

class GraphReasoningMixin:
    """图结构推理：识别子图里真实存在的拓扑模式，构造并校验可溯源的推理链。"""

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于图结构的推理：这是图RAG的智能之处
        不仅检索信息，还能进行逻辑推理
        """
        reasoning_chains = []

        try:
            # 1. 识别推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)

            # 2. 构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)

            # 3. 验证推理链的可信度
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)

            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains

        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []

    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别子图里【真实存在】的推理模式。

        与早期「返回写死的三个模式」不同，这里完全由子图实际包含的关系类型与节点结构决定：
        某个模式只有在它的证据关系确实出现时才会被列出，因此不同子图会得到不同的模式集合，
        后续 _build_reasoning_chain 才有真实拓扑可依据。
        """
        if subgraph is None:
            return []

        rel_types: Set[str] = {str(r.get("type") or "") for r in (subgraph.relationships or [])}
        node_labels: Set[str] = set()
        for node in list(subgraph.central_nodes or []) + list(subgraph.connected_nodes or []):
            node_labels.update(node.get("labels") or [])

        patterns: List[str] = []

        # 组成关系：菜谱 → 食材，是烹饪知识最基本的推理依据
        if "REQUIRES" in rel_types or ("Recipe" in node_labels and "Ingredient" in node_labels):
            patterns.append("组成关系")
        # 时序关系：步骤之间先后顺序
        if "NEXT_STEP" in rel_types or "CONTAINS_STEP" in rel_types or "CookingStep" in node_labels:
            patterns.append("时序关系")
        # 分类归属：菜谱 → 分类，支撑「同类菜品」推理
        if "BELONGS_TO_CATEGORY" in rel_types or "Category" in node_labels:
            patterns.append("分类归属")
        # 相似关系：显式 SIMILAR 边
        if "SIMILAR" in rel_types:
            patterns.append("相似关系")
        # 技法共性：同工具 / 同技法，跨菜谱的精细连通依据
        if "USES_SAME_METHOD" in rel_types or "USES_SAME_TOOL" in rel_types:
            patterns.append("技法共性")
        # 食材替代：同一子图内存在同类食材（如多个「调料」）才成立
        if self._has_sibling_ingredients(subgraph):
            patterns.append("食材替代")
        # 难度分层
        if "DIFFICULTY_LEVEL" in rel_types or "DifficultyLevel" in node_labels:
            patterns.append("难度分层")
        # 概念归类
        if "HAS_CONCEPT_TYPE" in rel_types or "ConceptType" in node_labels:
            patterns.append("概念归类")

        return patterns

    def _has_sibling_ingredients(self, subgraph: KnowledgeSubgraph) -> bool:
        """判断子图内是否存在「同类食材」（用于素材替代推理）。

        烹饪场景下最有价值的推理之一就是「缺某样调料时用什么替代」，
        它的图结构依据就是两个食材节点挂到同一个 Category 上。
        """
        if subgraph is None:
            return False
        category_to_ingredients: Dict[str, Set[str]] = defaultdict(set)
        for node in (subgraph.connected_nodes or []):
            labels = set(node.get("labels") or [])
            if "Ingredient" not in labels:
                continue
            category = str((node.get("properties") or {}).get("category") or "")
            if category:
                category_to_ingredients[category].add(str(node.get("name") or ""))
        return any(len(items) >= 2 for items in category_to_ingredients.values())

    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """按推理模式，用子图里的真实节点与边构造一条可读、可溯源的推理链。

        设计原则：每条链都必须【引用具体实体名】并【说明依据的关系类型】，
        这样它才能作为证据喂给生成模型；返回 None 表示该模式在当前子图上证据不足，不应产出噪声推理。
        """
        if subgraph is None:
            return None

        central_names = [str(n.get("name") or "") for n in (subgraph.central_nodes or []) if n.get("name")]
        central_text = "、".join(f"「{n}」" for n in central_names[:3]) if central_names else "该实体"

        # 按标签与关系类型把节点/边分好类，供各模式取用
        ingredients = [n for n in (subgraph.connected_nodes or []) if "Ingredient" in (n.get("labels") or [])]
        steps = [n for n in (subgraph.connected_nodes or []) if "CookingStep" in (n.get("labels") or [])]
        categories = [n for n in (subgraph.connected_nodes or []) if "Category" in (n.get("labels") or [])]
        recipes = [n for n in (subgraph.connected_nodes or []) if "Recipe" in (n.get("labels") or [])]
        difficulties = [n for n in (subgraph.connected_nodes or []) if "DifficultyLevel" in (n.get("labels") or [])]

        try:
            if pattern == "组成关系":
                if not ingredients:
                    return None
                by_category: Dict[str, List[str]] = defaultdict(list)
                for item in ingredients:
                    category = str((item.get("properties") or {}).get("category") or "未分类")
                    by_category[category].append(str(item.get("name") or ""))
                groups = []
                for category, names in sorted(by_category.items(), key=lambda kv: -len(kv[1]))[:3]:
                    groups.append(f"{category}类（{'、'.join([n for n in names if n][:4])}）")
                return (f"{central_text}的构成关系：共关联 {len(ingredients)} 种食材，"
                        f"按类别可分为 {'；'.join(groups)}。依据关系类型 REQUIRES。")

            if pattern == "时序关系":
                if not steps:
                    return None

                # 必须先按「所属菜谱」把步骤分组，否则会把多道菜的步骤混排成一条假工序。
                # 实测教训：一次子图里可能同时含 5 道川菜的步骤，它们的步骤名都叫「步骤1/步骤2…」，
                # 若不分组就会输出「步骤1 → 步骤2 → 步骤2 → 步骤2 → 步骤3」这种看似工序、实为串台的序列。
                owner_of_step: Dict[str, str] = {}
                for rel in (subgraph.relationships or []):
                    if str(rel.get("type") or "") in ("CONTAINS_STEP", "NEXT_STEP"):
                        start_id, end_id = str(rel.get("start_id") or ""), str(rel.get("end_id") or "")
                        if start_id and end_id and end_id not in owner_of_step:
                            owner_of_step[end_id] = start_id

                grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for step in steps:
                    grouped[owner_of_step.get(str(step.get("id") or ""), "__unowned__")].append(step)

                def _step_order(step: Dict[str, Any]) -> float:
                    """按 stepNumber 排序；缺失/非数值的排在最后。"""
                    value = (step.get("properties") or {}).get("stepNumber")
                    return value if isinstance(value, (int, float)) else 999.0

                # 优先展示有明确归属菜谱的组；无归属的步骤组放最后，并如实说明它无法确定工序
                sequences: List[str] = []
                for owner_id, group in sorted(
                    grouped.items(), key=lambda kv: (kv[0] == "__unowned__", -len(kv[1]))
                ):
                    if len(sequences) >= 3:
                        break
                    ordered = sorted(group, key=_step_order)
                    names = [str(s.get("name") or "") for s in ordered[:8]]
                    seq = " → ".join([n for n in names if n])
                    if not seq:
                        continue
                    if owner_id == "__unowned__":
                        sequences.append(f"{len(group)} 个步骤（未关联到具体菜谱，无法确定先后工序）：{seq}")
                    else:
                        owner_name = next(
                            (str(n.get("name") or "") for n in (subgraph.central_nodes or [])
                             if str(n.get("id") or "") == owner_id),
                            "",
                        )
                        prefix = f"「{owner_name}」" if owner_name else "某道菜"
                        sequences.append(f"{prefix}共 {len(group)} 步，依序为：{seq}")

                if not sequences:
                    return None
                return ("制作流程：" + "；".join(sequences) + "。依据关系类型 CONTAINS_STEP / NEXT_STEP。")

            if pattern == "分类归属":
                if not categories:
                    return None
                names = [str(c.get("name") or "") for c in categories][:5]
                return (f"{central_text}在分类体系中归属于 {'、'.join([n for n in names if n])}，"
                        f"因此可与同分类下的其他条目互为参照。依据关系类型 BELONGS_TO_CATEGORY。")

            if pattern == "相似关系":
                if not recipes:
                    return None
                names = [str(r.get("name") or "") for r in recipes][:5]
                return (f"{central_text}与 {'、'.join([n for n in names if n])} 之间存在显式相似关系，"
                        f"可作为同类替代或对比的候选。依据关系类型 SIMILAR。")

            if pattern == "技法共性":
                methods = _dedup_preserve_order([
                    str((r.get("properties") or {}).get("method") or (r.get("properties") or {}).get("tool") or "")
                    for r in (subgraph.relationships or [])
                    if str(r.get("type") or "") in ("USES_SAME_METHOD", "USES_SAME_TOOL")
                ])
                methods = [m for m in methods if m]
                if not methods:
                    return None
                return (f"{central_text}在制作技法上与其它菜谱存在共性，共享的技法/工具包括："
                        f"{'、'.join(methods[:5])}。依据关系类型 USES_SAME_METHOD / USES_SAME_TOOL。")

            if pattern == "食材替代":
                by_category: Dict[str, List[str]] = defaultdict(list)
                for item in ingredients:
                    category = str((item.get("properties") or {}).get("category") or "")
                    name = str(item.get("name") or "")
                    if category and name:
                        by_category[category].append(name)
                substitutes = [
                    (category, names) for category, names in by_category.items() if len(names) >= 2
                ]
                if not substitutes:
                    return None
                parts = [
                    f"缺少{category}类食材时，可在 {'、'.join(names[:4])} 之间择一替代"
                    for category, names in sorted(substitutes, key=lambda kv: -len(kv[1]))[:3]
                ]
                return (f"{central_text}的食材替代参考：" + "；".join(parts) + "。依据同一 Category 下食材的可互换性。")

            if pattern == "难度分层":
                if not difficulties:
                    return None
                names = [str(d.get("name") or "") for d in difficulties][:5]
                return (f"{central_text}处于 {'、'.join([n for n in names if n])} 难度层级，"
                        f"可用于按难度筛选或排序同类菜谱。依据关系类型 DIFFICULTY_LEVEL。")

            if pattern == "概念归类":
                concepts = [str(n.get("name") or "") for n in (subgraph.connected_nodes or [])
                            if "ConceptType" in (n.get("labels") or [])]
                concepts = [c for c in concepts if c]
                if not concepts:
                    return None
                return (f"{central_text}在概念层被归类为 {'、'.join(concepts[:3])}。"
                        f"注意该关系覆盖范围极广，推理时区分度有限。依据关系类型 HAS_CONCEPT_TYPE。")

            return None
        except Exception as e:
            logger.warning(f"构造推理链失败（模式={pattern}）: {e}")
            return None

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """对推理链做可信度校验与排序（替代早期「只截前三条」的假校验）。

        校验口径（只依赖文本自身与查询，无需隐藏状态）：
          - 与查询的词法重合度：链条若完全答非所问则应当降权；
          - 具体性：是否引用了具名实体（「…」）、是否带数量或步骤号 —— 越具体越可能是真推理而非套话；
        然后去重、按综合分降序、截断到固定条数。
        """
        if not chains:
            return []

        query_bigrams = _char_bigrams(query or "")
        scored: List[Tuple[float, str]] = []
        seen: Set[str] = set()

        for chain in chains:
            text = (chain or "").strip()
            if not text:
                continue
            signature = re.sub(r"\s+", "", text)
            if signature in seen:
                continue
            seen.add(signature)

            overlap = _containment(query_bigrams, _char_bigrams(text)) if query_bigrams else 0.0

            concreteness = 0.0
            if "「" in text and "」" in text:
                concreteness += 0.5          # 引用了具名实体
            if re.search(r"\d", text):
                concreteness += 0.3          # 含数量/序号等可核验细节
            if len(text) >= 20:
                concreteness += 0.2          # 足够长才承载得了推理

            score = 0.6 * overlap + 0.4 * concreteness
            scored.append((score, text))

        scored.sort(key=lambda item: -item[0])
        # 最多返回 5 条：推理链是给生成模型的辅助证据，过多反而稀释上下文
        return [text for _, text in scored[:5]]
