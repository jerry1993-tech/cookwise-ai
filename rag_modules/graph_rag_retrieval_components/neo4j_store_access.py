"""图 RAG 的 Neo4j 访问层：连接生命周期、只读 Cypher、节点/关系序列化、度数补齐与关系去重。"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase

from .shared_constants_helpers_models import (
    _DEFAULT_RELATION_WEIGHT,
    _HUB_DEGREE_THRESHOLD,
    _MEANINGFUL_LABELS,
    _PLACEHOLDER_NAMES,
    _PLACEHOLDER_NODE_IDS,
    _RELATION_WEIGHTS,
    _dedup_preserve_order,
)

logger = logging.getLogger(__name__)


class GraphStoreMixin:
    """图存储访问：连接生命周期、只读 Cypher、节点/关系序列化、度数补齐与关系去重。

    所有 Neo4j 访问都收口在这里，其他组件只通过 self 使用这些能力。"""

    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")

        # 连接Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            # 【必须把 driver 清空，否则降级形同虚设】
            # neo4j 的 driver 是【惰性】的：GraphDatabase.driver(...) 只构造对象，
            # 不建立任何真实连接，所以哪怕 Neo4j 完全没启动，self.driver 也不是 None。
            # 而 _is_driver_ready() 判的就是 `driver is not None` —— 于是"连接失败"之后
            # 每个查询点仍然以为图库可用，一路走到 Cypher 执行才炸，最后悄悄退回降级。
            # 结果是最需要被观测的场景（图库全挂）在日志与 metadata 里都不可见，
            # 测试断言"返回非空"也照样通过。这里显式关闭并置空，
            # 让 _is_driver_ready() 如实返回 False，各查询点一致走降级分支。
            self._discard_driver()
            return

        # 预热：构建实体和关系索引
        self._build_graph_index()

    def _discard_driver(self) -> None:
        """关闭并丢弃当前 driver（连接失败或初始化异常时的统一收口）。

        之所以要"关闭 + 置空"两步都做：
          - 置空是为了让 _is_driver_ready() 如实返回 False，后续查询一致走降级；
          - 关闭是为了不泄漏连接池（惰性 driver 被置空前可能已经建过池）。
        任何一步失败都不应影响主流程，故整体吞异常只记日志。
        """
        driver = getattr(self, "driver", None)
        self.driver = None
        if driver is not None:
            try:
                driver.close()
            except Exception as e:
                logger.warning(f"关闭 Neo4j driver 失败（忽略）: {e}")

    def _is_driver_ready(self) -> bool:
        """判断 Neo4j 驱动是否可用。

        所有图操作的统一前置检查：连接缺失时一律安全降级为空结果，
        绝不让图库的抖动打挂整条 RAG 链路。

        注意：这里【只能】判断 driver 是否存在，不能在这里做 verify_connectivity()
        之类的真实探测 —— 本方法被每个查询点高频调用，加一次网络往返会让整体延迟翻倍。
        真正的可达性由 initialize() 负责：连不上时它会调 _discard_driver() 把 driver
        置空，因此"driver 非 None"等价于"initialize 时探测成功过"。
        driver 本身是惰性对象，绝不能仅凭"对象非 None"就认为图库可用（见 initialize 的说明）。
        """
        return getattr(self, "driver", None) is not None

    def _run_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """执行只读 Cypher 并返回字典列表。

        设计意图：把「建会话 / 执行 / 记录耗时与条数 / 吞掉异常并记日志」这四件事收口到一处，
        避免每个查询点各写一份 try/except 导致日志风格与降级行为不一致。

        Args:
            cypher: Cypher 语句（只读查询；本模块不通过它做写操作）
            params: 查询参数。凡是用户/LLM 产出的值都必须走参数，绝不做字符串拼接
            limit: 可选的返回条数上限（Python 侧截断，作为 Cypher 内 LIMIT 之外的第二道保险）

        Returns:
            记录字典列表；任何异常都会被记录并降级为 []
        """
        if not self._is_driver_ready():
            logger.error("Neo4j连接未建立，无法执行查询")
            return []

        import time as _time  # 局部导入：仅在真正查询时才需要计时，避免模块级引入无关依赖

        started = _time.time()
        try:
            with self.driver.session() as session:
                records = [dict(record) for record in session.run(cypher, params or {})]
            elapsed_ms = (_time.time() - started) * 1000.0
            if limit is not None:
                records = records[:limit]
            # 慢查询告警：图遍历一旦退化到秒级，通常意味着少了 hub 剪枝或标签过滤
            if elapsed_ms > 2000:
                logger.warning(f"Cypher 慢查询 {elapsed_ms:.0f}ms，返回 {len(records)} 条：{cypher.strip()[:160]}")
            else:
                logger.debug(f"Cypher 完成 {elapsed_ms:.0f}ms，返回 {len(records)} 条")
            return records
        except Exception as e:
            logger.error(f"Cypher 执行失败: {e} | 语句片段: {cypher.strip()[:200]}")
            return []

    def _node_to_dict(self, node) -> Dict[str, Any]:
        """把 neo4j Node 转成本模块内部统一使用的节点字典。

        统一字段约定（下游 _build_path_description / 排序 / 子图组装都依赖它）：
            id / node_id  节点业务主键（图谱里的 nodeId 属性）
            name          节点名称，缺失时退化为 nodeId，保证永不为空字符串
            labels        标签列表
            label         主标签（便于快速判断节点类型）
            properties    全部原始属性
            degree        节点度数，由 _attach_degrees 批量补齐（默认 0）
        """
        try:
            properties = dict(node)
        except Exception:
            properties = {}
        try:
            labels = list(node.labels)
        except Exception:
            labels = []

        node_id = str(properties.get("nodeId") or "")
        return {
            "id": node_id,
            "node_id": node_id,
            "name": str(properties.get("name") or node_id or "未知节点"),
            "labels": labels,
            "label": labels[0] if labels else "",
            "properties": properties,
            "degree": 0,
        }

    def _rel_to_dict(self, rel) -> Dict[str, Any]:
        """把 neo4j Relationship 转成本模块内部统一使用的关系字典。

        注意：关系类型必须用驱动的 `rel.type` 属性读取。
        早期实现里的 `type(rel).__name__` 只会返回 Python 类名 'Relationship'，永远拿不到真实类型——这是个真实缺陷。
        """
        try:
            properties = dict(rel)
        except Exception:
            properties = {}

        try:
            rel_type = str(rel.type)
        except Exception:
            # 极端情况下退回导入时保留的原始类型编码
            rel_type = str(properties.get("originalType") or "RELATED")

        start_id, end_id = "", ""
        try:
            start_id = str(rel.start_node.get("nodeId") or "")
            end_id = str(rel.end_node.get("nodeId") or "")
        except Exception:
            pass

        return {
            "type": rel_type,
            "rel_type": rel_type,
            "properties": properties,
            "start_id": start_id,
            "end_id": end_id,
            "weight": _RELATION_WEIGHTS.get(rel_type, _DEFAULT_RELATION_WEIGHT),
        }

    def _attach_degrees(self, nodes: List[Dict[str, Any]]) -> None:
        """一次性批量补齐一组节点的度数（原地写回 node["degree"]）。

        hub 剪枝与结构评分都需要度数，但逐个节点查一次度数会产生 N 次往返，
        所以这里把节点收集起来批量取回。

        性能要点：写成「一次全图扫描 + IN 过滤」，而不是 `UNWIND $ids AS nid MATCH (n {nodeId: nid})`。
        原因：nodeId 上的唯一索引只建在 Recipe / Ingredient / CookingStep 三个标签上，
        而【不带标签】的 `MATCH (n {nodeId: ...})` 无法命中任何一个标签索引，会退化成「每个 id 扫一次全图」。
        实测 200 个 id：UNWIND 写法 2517ms vs 单次扫描 101ms，相差 25 倍。

        【NULL nodeId 的补齐】Category / ConceptType 的 nodeId 全是 NULL，无法用 nodeId 反查
        （它们会静默拿到度数 0 → _is_hub 永远返回 False → hub 剪枝对它们完全失效）。
        而修好 NULL 过滤后 Category 会重新进入路径，其中 '调料'(1556)、'蔬菜'(605) 正是必须被剪掉的 hub。
        所以这里对 nodeId 为空的节点改用「标签 + 名称」二次反查。
        """
        if not self._is_driver_ready():
            return
        ids = _dedup_preserve_order([n.get("id") for n in nodes if n.get("id")])
        # nodeId 为空的节点（Category/ConceptType）：收集它们的名字做二次反查
        null_named = _dedup_preserve_order([
            str(n.get("name")) for n in nodes if not n.get("id") and n.get("name")
        ])

        degree_map: Dict[str, int] = {}

        if ids:
            rows = self._run_cypher(
                """
                MATCH (n)
                WHERE n.nodeId IN $ids
                RETURN n.nodeId AS id, COUNT { (n)--() } AS degree
                """,
                {"ids": ids},
            )
            degree_map.update({str(r.get("id")): int(r.get("degree") or 0) for r in rows})

        if null_named:
            rows = self._run_cypher(
                """
                MATCH (n)
                WHERE n.nodeId IS NULL AND n.name IN $names
                RETURN n.name AS nm, COUNT { (n)--() } AS degree
                """,
                {"names": null_named},
            )
            degree_map.update({str(r.get("nm")): int(r.get("degree") or 0) for r in rows})

        for node in nodes:
            key = str(node.get("id") or "") or str(node.get("name") or "")
            node["degree"] = degree_map.get(key, 0)

    def _is_hub(self, node: Dict[str, Any]) -> bool:
        """判断节点是否为需要剪枝的 hub（度数超过阈值）。"""
        return int(node.get("degree") or 0) > _HUB_DEGREE_THRESHOLD

    def _is_meaningful_node(self, node: Dict[str, Any]) -> bool:
        """判断节点是否属于业务语义节点（过滤统计/层次节点与占位脏节点）。"""
        node_id = str(node.get("id") or "")
        # 占位脏节点（nodeId 200000000/300000000，name 为「菜谱」「食材」）一律排除
        if node_id in _PLACEHOLDER_NODE_IDS:
            return False
        labels = set(node.get("labels") or [])
        if not (labels & set(_MEANINGFUL_LABELS)):
            return False
        return str(node.get("name") or "") not in _PLACEHOLDER_NAMES

    def _node_identity_key(self, node: Dict[str, Any]) -> str:
        """返回节点在【子图组装阶段】的唯一标识，用于去重、排除与前沿传递。

        为什么不能直接用 node["id"] 做集合运算：
        实测本图谱里 Category / ConceptType / DifficultyLevel / RecipeCategory 这几类节点的
        nodeId 属性【全部为 NULL】，它们的 properties 里只有 name（如 {'name': '荤菜'}）。
        若直接拿 node["id"] 参与去重，这些节点会一起退化成空串 ""，引发两重故障：
          1. 丢失节点：`if not node["id"]: continue` 这类真值判断会把它们整个丢弃
             —— 宫保鸡丁 的子图里 Category 恒为 0，而 荤菜(deg=97) 就在一跳之外；
          2. 节点互相顶替：多个 Category 邻居被判为「重复」而只保留第一个
             —— 实测有 10 个菜谱挂了 2~3 个 Category（如 猪皮冻 = 下酒菜/凉菜/荤菜）。
        因此这里对没有 nodeId 的节点退回用 name 造一个带前缀的合成键。
        前缀 "@name:" 用于保证不与真实 nodeId 冲突（真实 nodeId 均为纯数字串）。

        注意：本键【仅用于子图内部的集合运算】，不会写回 node["id"]，
        所以 node["id"] 表示「图谱真实 nodeId」这一对外语义保持不变
        （无 nodeId 的节点其 id 仍为 ""，下游可据此判断该节点不可用 nodeId 反查）。

        Args:
            node: 节点字典，至少包含 id / node_id / name 之一

        Returns:
            非空字符串；若 id 与 name 均为空（脏节点）则返回 ""，调用方据此过滤
        """
        node_id = str(node.get("id") or node.get("node_id") or "").strip()
        if node_id:
            return node_id
        name = str(node.get("name") or "").strip()
        return f"@name:{name}" if name else ""

    def _append_relationship(self, relationships: List[Dict[str, Any]],
                             seen_rel: Set[Tuple[str, str, str]],
                             row: Dict[str, Any]) -> None:
        """把一行三元组里的关系追加进关系列表（按 类型+起点身份键+终点身份键 去重）。

        之所以把关系的记录从「邻居节点去重」里拆出来单独处理：
        节点去重（同一个邻居被多条行反复返回）不应该连带把关系也丢掉，
        否则会损失两类真实存在的边：
          - 平行边：同一对节点之间存在多种关系类型。实测本图谱有 376 对这样的节点，
            例如 Recipe→Ingredient 同时有 REQUIRES 与 USES_SAME_METHOD；
          - 跨种子边：多实体查询时，邻居 nbX 由种子 A 首次收录，
            种子 B→nbX 的那条边会在 B 的行上被「节点已存在」判掉，
            导致多实体子图丢失种子间的连通证据。

        【边签名为什么用身份键而不是裸 nodeId】
        裸 nodeId 作为签名的一部分时，无 nodeId 的端点（Category 等）一律退化成空串，
        于是 调料 / 荤菜 这类同级邻居会被判成「同一条边」而只保留一条。
        这里改用 startNode/endNode 的节点身份键（nodeId，缺失时退回 @name:xxx）：
        由于 startNode/endNode 由边自身的方向决定，同一条边无论从哪端遍历进来
        都会算出同一个签名，天然方向归一，不会重复计数。

        Args:
            relationships: 关系列表（原地追加）
            seen_rel: 已收录关系的签名集合（原地更新）
            row: 单行查询结果，需含 rel_type / rel_props，以及
                 rel_start / rel_start_name / rel_end / rel_end_name 四个端点列
        """
        rel_type = str(row.get("rel_type") or "RELATED")
        rel_start = str(row.get("rel_start") or "")
        rel_end = str(row.get("rel_end") or "")
        # 端点身份键：nodeId 为空时用端点的 name 顶替，保证无 nodeId 的端点也能区分
        start_key = self._node_identity_key({"id": rel_start, "name": row.get("rel_start_name")})
        end_key = self._node_identity_key({"id": rel_end, "name": row.get("rel_end_name")})
        rel_signature = (rel_type, start_key, end_key)
        if rel_signature in seen_rel:
            return
        seen_rel.add(rel_signature)
        relationships.append({
            "type": rel_type,
            "rel_type": rel_type,
            "properties": dict(row.get("rel_props") or {}),
            # start_id / end_id 保持"图谱真实 nodeId"的对外语义不变（可能为空串）
            "start_id": rel_start,
            "end_id": rel_end,
            # 端点名一并落库：一是给上层溯源用，二是让 _dedupe_relationships 能在
            # 【跨阶段合并之后】重新算出与这里完全一致的边签名（无 nodeId 的端点
            # 必须靠 name 才能区分，只留 nodeId 会算不出来）。
            "start_name": str(row.get("rel_start_name") or ""),
            "end_name": str(row.get("rel_end_name") or ""),
            "weight": _RELATION_WEIGHTS.get(rel_type, _DEFAULT_RELATION_WEIGHT),
        })

    def _dedupe_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对【多阶段合并后】的关系列表做跨阶段去重，保持首次出现的顺序。

        为什么必须补这一步：`_assemble_from_rows` 的 seen_rel 是"调用内局部集合"
        （不能改成参数，签名不可变），而 `_collect_subgraph` 的两阶段抽取会各自调用它
        再把结果 extend 到一起。当阶段二的遍历边有一端是本次查询的【种子】时
        （种子在阶段一已被 (seed)-[r]-(nb) 记录过），hop2 行会因为在 current_ids 中
        命中"节点已存在、只补边"分支而再次追加同一条边 —— 新调用看不到阶段一写过的边。

        实测：extract_knowledge_subgraph(GraphQuery(ENTITY_RELATION, ['花椒'],
        max_depth=2, max_nodes=50)) 得到 52 条关系，其中唯一签名只有 49 条；
        全量 322 菜谱 x {10,20,50} 有 227/966 例出现跨阶段重复。
        重复边会把 relationship_count 与 density 一起抬高（实测最高 +19%），
        而这两个指标会进入 _subgraph_struct_score 的排序分，并写进喂给 LLM 的子图描述。

        签名口径与 `_append_relationship` 严格一致：类型 + 起点身份键 + 终点身份键
        （身份键 = nodeId，缺失时退回 @name:xxx），因此无 nodeId 的端点（Category 等）
        也能正确参与比对。

        Args:
            relationships: 可能含跨阶段重复的关系列表

        Returns:
            去重后的新列表（不修改入参）
        """
        deduped: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            rel_type = str(rel.get("type") or rel.get("rel_type") or "RELATED")
            start_key = self._node_identity_key({
                "id": rel.get("start_id"), "name": rel.get("start_name"),
            })
            end_key = self._node_identity_key({
                "id": rel.get("end_id"), "name": rel.get("end_name"),
            })
            signature = (rel_type, start_key, end_key)
            if not start_key and not end_key:
                # 两端都无法标识的脏关系，保留但不参与去重（宁可留一条可疑边，
                # 也不要在信息不足时静默丢边）
                deduped.append(rel)
                continue
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(rel)
        dropped = len(relationships) - len(deduped)
        if dropped > 0:
            logger.info(f"关系跨阶段去重：移除 {dropped} 条重复边（{len(relationships)} -> {len(deduped)}）")
        return deduped

    def close(self):
        """关闭资源连接。

        【为什么必须置空 driver】_is_driver_ready() 的判据就是 `driver is not None`，
        而它自身的 docstring 声明的契约是"driver 非 None 等价于 initialize 时探测成功过"。
        只 close 不置空时该契约被破坏：close() 之后 _is_driver_ready() 仍返回 True，
        5 个查询点会继续走 Cypher 分支，逐条记 ERROR "Cypher 执行失败: Driver closed"，
        关闭后的行为既不可观测又与"连接不可用"的降级语义不一致。
        这里统一收口到 _discard_driver()，与 initialize() 失败路径口径完全一致（关闭 + 置空）。
        """
        if hasattr(self, 'driver') and self.driver:
            self._discard_driver()
            logger.info("图RAG检索系统已关闭")
