"""
GraphRAGRetrieval 检索能力验证脚本（最小可执行）

设计意图
--------
与 ``tests/test_graph_rag.py``（演示型脚本，只打印不校验）互补，本脚本是**校验型**脚本：
它对 ``GraphRAGRetrieval`` 的每一个核心能力施加可判定的不变量断言，并给出 PASS/FAIL 汇总，
最终以退出码表达结果，便于 CI 或人工回归。

覆盖的七个能力点，与类文档字符串声明的五点核心特点一一对应：
1. 查询意图理解        -> understand_graph_query
2. 动态查询规划        -> adaptive_query_planning / _analyze_query_complexity
3. 多跳图遍历          -> multi_hop_traversal（覆盖 MULTI_HOP / ENTITY_RELATION / PATH_FINDING）
4. 子图提取            -> extract_knowledge_subgraph
5. 图结构推理          -> graph_structure_reasoning
6. 结果组织与排序      -> _rank_by_graph_relevance（通过 graph_rag_search 间接校验）
7. 降级与防御          -> 未初始化 driver 时必须安全返回空结果而不是抛异常

前置条件
--------
- Neo4j 已启动且已导入菜谱图谱（docker compose up -d，见 docker-compose.yml）
- 项目根目录 ``.env`` 中已配置 OPENAI_API_KEY 与 OPENAI_BASE_URL

运行方式
--------
    python tests/test_graph_rag_verification.py

退出码：0 = 全部通过（允许有 WARN）；1 = 存在 FAIL。
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, List, Tuple

from dotenv import load_dotenv

# 先加载 .env（config.py 在导入时即读取环境变量），再修正模块搜索路径以便从任意位置运行
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH if _ENV_PATH.exists() else None)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import GraphRAGConfig  # noqa: E402
from rag_modules.graph_rag_retrieval import (  # noqa: E402
    GraphQuery,
    QueryType,
    GraphRAGRetrieval,
)


# ============================ 结果收集器 ============================


class _Reporter:
    """极简的断言收集器：记录 PASS / FAIL / WARN，并负责最终汇总与退出码。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: List[str] = []
        self.warned: List[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        """记录一条断言结果；返回断言是否成立，便于调用方做后续分支。"""
        if ok:
            self.passed += 1
            print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed.append(f"{label} :: {detail}")
            print(f"  [FAIL] {label}  {detail}")
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        """记录一条非致命告警（例如依赖外部 LLM 因此结果不稳定）。"""
        self.warned.append(f"{label} :: {detail}")
        print(f"  [WARN] {label}  {detail}")

    def summary(self) -> int:
        """打印汇总并返回进程退出码。"""
        print("\n" + "=" * 64)
        print(f"验证汇总：PASS {self.passed} 项 | FAIL {len(self.failed)} 项 | WARN {len(self.warned)} 项")
        for msg in self.failed:
            print(f"  FAIL -> {msg}")
        for msg in self.warned:
            print(f"  WARN -> {msg}")
        print("=" * 64)
        return 1 if self.failed else 0


# ============================ 工具函数 ============================

# 允许出现在结果中的真实业务标签；垃圾/统计节点标签一旦出现即视为过滤失效
_MEANINGFUL_LABELS = {"Recipe", "Ingredient", "CookingStep", "Category", "ConceptType", "DifficultyLevel"}
# 图谱中的占位脏节点（见 data/cypher/nodes.csv 的 200000000 / 300000000）
_PLACEHOLDER_NAMES = {"菜谱", "食材"}


def _build_llm_client():
    """构造 OpenAI 兼容客户端；缺少 key 时返回 None 以便脚本降级为纯离线校验。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))


def _node_names(path) -> List[str]:
    """把 GraphPath / 子图节点列表里的名字安全地取出来，用于可读性与断言。"""
    if isinstance(path, list):
        nodes = path
    else:
        nodes = getattr(path, "nodes", []) or []
    return [str(n.get("name", "")) for n in nodes if isinstance(n, dict)]


def _section(title: str) -> None:
    print("\n" + "-" * 64)
    print(f"# {title}")
    print("-" * 64)


# ============================ 各能力点的验证 ============================


def verify_intent_understanding(module: GraphRAGRetrieval, rep: _Reporter) -> None:
    """
    验证「查询意图理解」：自然语言 -> GraphQuery。

    不变量：
      - query_type 必须落在 QueryType 枚举内（LLM 返回非法值时也必须被兜底）
      - max_depth 必须被钳制在 [1, 3]（防止 LLM 返回 10 导致 Cypher 爆炸）
      - 对明确的实体型查询，source_entities 必须非空（否则整条检索链拿不到起点）
    """
    _section("能力1：查询意图理解 understand_graph_query")

    cases = [
        ("鸡肉和胡萝卜能一起做菜吗？", QueryType.ENTITY_RELATION),
        ("宫保鸡丁需要哪些食材？", None),
        ("川菜有什么特色菜", None),
        ("和宫保鸡丁类似的菜有哪些？", None),
    ]

    for query, _expected in cases:
        try:
            gq = module.understand_graph_query(query)
        except Exception as exc:  # 任何异常都是实现缺陷：该方法是检索入口，必须自我兜底
            rep.check(False, f"understand_graph_query({query!r})", f"抛出异常 {type(exc).__name__}: {exc}")
            continue

        rep.check(isinstance(gq, GraphQuery), f"返回类型为 GraphQuery（{query!r}）", type(gq).__name__)
        rep.check(
            isinstance(gq.query_type, QueryType),
            f"query_type 合法（{query!r}）",
            str(getattr(gq.query_type, "value", gq.query_type)),
        )
        rep.check(
            1 <= int(gq.max_depth) <= 3,
            f"max_depth 被钳制在 [1,3]（{query!r}）",
            f"max_depth={gq.max_depth}",
        )
        rep.check(
            bool(gq.source_entities),
            f"source_entities 非空（{query!r}）",
            f"{gq.source_entities}",
        )
        print(f"    -> 识别为 {gq.query_type.value}，实体={gq.source_entities}，深度={gq.max_depth}")


def verify_query_planning(module: GraphRAGRetrieval, rep: _Reporter) -> None:
    """
    验证「动态查询规划」：复杂度评分应随查询复杂度单调变化，并据此产出 1~2 个计划。

    不变量：
      - _analyze_query_complexity 返回值恒在 [0, 1]
      - 简单查询与复杂查询的复杂度评分应有可观测差异（否则规划退化）
      - adaptive_query_planning 始终返回非空计划列表，且每个计划 max_depth 在 [1,3]
    """
    _section("能力2：动态查询规划 adaptive_query_planning")

    simple = "宫保鸡丁"
    complex_q = "鸡肉配什么蔬菜比较好，为什么这些搭配合理，以及和川菜中类似做法有什么区别？"

    try:
        c_simple = module._analyze_query_complexity(simple)
        c_complex = module._analyze_query_complexity(complex_q)
    except Exception as exc:
        rep.check(False, "_analyze_query_complexity 可调用", f"{type(exc).__name__}: {exc}")
        return

    rep.check(0.0 <= c_simple <= 1.0, "复杂度评分落在 [0,1]（简单查询）", f"{c_simple:.3f}")
    rep.check(0.0 <= c_complex <= 1.0, "复杂度评分落在 [0,1]（复杂查询）", f"{c_complex:.3f}")
    rep.check(
        c_complex > c_simple,
        "复杂度评分能区分简单/复杂查询",
        f"simple={c_simple:.3f} vs complex={c_complex:.3f}",
    )

    for label, q in (("简单查询", simple), ("复杂查询", complex_q)):
        try:
            plans = module.adaptive_query_planning(q)
        except Exception as exc:
            rep.check(False, f"adaptive_query_planning({label})", f"{type(exc).__name__}: {exc}")
            continue
        rep.check(bool(plans), f"规划结果非空（{label}）", f"计划数={len(plans)}")
        rep.check(
            all(1 <= int(p.max_depth) <= 3 for p in plans),
            f"所有计划 max_depth 在 [1,3]（{label}）",
            f"{[p.max_depth for p in plans]}",
        )
        rep.check(
            all(isinstance(p.query_type, QueryType) for p in plans),
            f"所有计划 query_type 合法（{label}）",
            f"{[p.query_type.value for p in plans]}",
        )
        print(f"    -> {label} 复杂度={module._analyze_query_complexity(q):.3f}，产出 {len(plans)} 个计划：{[p.query_type.value for p in plans]}")


def verify_multi_hop_traversal(module: GraphRAGRetrieval, rep: _Reporter) -> None:
    """
    验证「多跳图遍历」：覆盖三种查询类型分支。

    不变量：
      - 返回 List[GraphPath]，且每个 GraphPath 结构完整（nodes 非空、path_length >= 1、分数为有限数）
      - 关系类型必须是真实 Cypher 类型名（不能是 Python 类名 'Relationship'）
      - 结果的 relevance_score 需为有限值（排序的依据必须可靠）
      - 结果中不得混入 GraphStatistics 等垃圾节点
    """
    _section("能力3：多跳图遍历 multi_hop_traversal")

    plans = [
        ("MULTI_HOP", GraphQuery(query_type=QueryType.MULTI_HOP, source_entities=["宫保鸡丁"],
                                 target_entities=["蔬菜类食材"], relation_types=["REQUIRES"], max_depth=2)),
        ("ENTITY_RELATION", GraphQuery(query_type=QueryType.ENTITY_RELATION, source_entities=["宫保鸡丁"],
                                       relation_types=["REQUIRES"], max_depth=2)),
        ("PATH_FINDING", GraphQuery(query_type=QueryType.PATH_FINDING, source_entities=["宫保鸡丁"],
                                    target_entities=["黄瓜"], max_depth=3)),
    ]

    for label, gq in plans:
        try:
            paths = module.multi_hop_traversal(gq)
        except Exception as exc:
            rep.check(False, f"multi_hop_traversal({label}) 不抛异常", f"{type(exc).__name__}: {exc}")
            continue

        if not rep.check(isinstance(paths, list), f"返回 list（{label}）", type(paths).__name__):
            continue
        if not paths:
            rep.check(False, f"检索到路径（{label}）", "返回空列表，该分支疑似仍为空实现")
            continue

        rep.check(True, f"检索到路径（{label}）", f"{len(paths)} 条")

        bad_struct = [p for p in paths if not getattr(p, "nodes", None)]
        rep.check(not bad_struct, f"所有路径 nodes 非空（{label}）", f"异常路径数={len(bad_struct)}")

        finite = [p for p in paths if isinstance(getattr(p, "relevance_score", None), (int, float))]
        rep.check(len(finite) == len(paths), f"所有路径 relevance_score 为数值（{label}）",
                  f"{len(finite)}/{len(paths)}")

        rel_types = {r.get("type") for p in paths for r in (getattr(p, "relationships", None) or [])}
        rep.check(
            "Relationship" not in rel_types,
            f"关系类型不是 Python 类名（{label}）",
            f"发现的关系类型={sorted(t for t in rel_types if t)[:6]}",
        )

        names = {n for p in paths for n in _node_names(p)}
        leaked = names & _PLACEHOLDER_NAMES
        rep.check(not leaked, f"无占位脏节点混入（{label}）", f"泄漏={sorted(leaked)}")

        sample = paths[0]
        print(f"    -> {label}: {len(paths)} 条路径，样例：")
        print(f"       {' -> '.join(_node_names(sample))[:160]}")
        print(f"       path_length={sample.path_length} relevance={sample.relevance_score}")


def verify_subgraph_extraction(module: GraphRAGRetrieval, rep: _Reporter) -> None:
    """
    验证「知识子图提取」。

    不变量：
      - central_nodes 非空（否则子图没有中心，后续推理无意义）
      - connected_nodes / relationships 有实质内容（这是"知识网络"的本体）
      - graph_metrics 必须包含可用的图指标键
      - 不得包含 GraphStatistics / RelationshipStatistics 等垃圾节点
    """
    _section("能力4：知识子图提取 extract_knowledge_subgraph")

    gq = GraphQuery(query_type=QueryType.SUBGRAPH, source_entities=["宫保鸡丁"], max_depth=2, max_nodes=60)

    try:
        sub = module.extract_knowledge_subgraph(gq)
    except Exception as exc:
        rep.check(False, "extract_knowledge_subgraph 不抛异常", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    rep.check(bool(getattr(sub, "central_nodes", None)), "central_nodes 非空",
              f"中心节点={_node_names(getattr(sub, 'central_nodes', []))}")
    rep.check(bool(getattr(sub, "connected_nodes", None)), "connected_nodes 非空",
              f"{len(getattr(sub, 'connected_nodes', []) or [])} 个")
    rep.check(bool(getattr(sub, "relationships", None)), "relationships 非空",
              f"{len(getattr(sub, 'relationships', []) or [])} 个")

    metrics = getattr(sub, "graph_metrics", {}) or {}
    rep.check(bool(metrics), "graph_metrics 非空", f"键={sorted(metrics)}")
    rep.check(
        any(k in metrics for k in ("node_count", "relationship_count", "density")),
        "graph_metrics 含标准图指标",
        f"键={sorted(metrics)}",
    )

    all_names = set(_node_names(getattr(sub, "central_nodes", []))) | set(_node_names(getattr(sub, "connected_nodes", [])))
    leaked = all_names & _PLACEHOLDER_NAMES
    rep.check(not leaked, "无占位脏节点混入子图", f"泄漏={sorted(leaked)}")

    junk = {"GraphStatistics", "RelationshipStatistics"}
    sub_labels = {
        lbl
        for n in (getattr(sub, "connected_nodes", []) or []) + (getattr(sub, "central_nodes", []) or [])
        for lbl in (n.get("labels") or [])
    }
    rep.check(not (sub_labels & junk), "无统计垃圾节点混入", f"泄漏={sorted(sub_labels & junk)}")

    # 子图自然语言描述必须包含真实节点名，否则它无法作为检索证据喂给 LLM
    desc = module._build_subgraph_description(sub)
    rep.check(bool(desc and len(desc.strip()) > 10), "_build_subgraph_description 产出有效文本",
              f"长度={len(desc or '')}")
    central_name = (_node_names(getattr(sub, "central_nodes", [])) or [""])[0]
    rep.check(central_name and central_name in desc, "子图描述包含中心实体名",
              f"中心={central_name!r}")

    print(f"    -> 子图规模：{len(sub.connected_nodes)} 节点 / {len(sub.relationships)} 关系，指标={metrics}")
    print(f"       描述片段：{desc[:150]}")


def verify_graph_reasoning(module: GraphRAGRetrieval, rep: _Reporter) -> None:
    """
    验证「图结构推理」：推理链必须是由真实子图拓扑推导出的、含真实节点名的可读文本。

    不变量：
      - 返回 List[str]
      - 推理链不得是 "基于X的推理链" 这类模板占位串
      - 推理链必须引用子图里真实存在的节点名（可溯源，才能作为证据）
      - 推理模式识别结果不得是写死的常量列表
    """
    _section("能力5：图结构推理 graph_structure_reasoning")

    gq = GraphQuery(query_type=QueryType.SUBGRAPH, source_entities=["宫保鸡丁"], max_depth=2, max_nodes=60)
    sub = module.extract_knowledge_subgraph(gq)
    query = "为什么宫保鸡丁要用这些食材？"

    try:
        patterns = module._identify_reasoning_patterns(sub)
        rep.check(isinstance(patterns, list), "_identify_reasoning_patterns 返回 list", str(type(patterns).__name__))
        print(f"    -> 识别到的推理模式：{patterns}")
    except Exception as exc:
        rep.check(False, "_identify_reasoning_patterns 可调用", f"{type(exc).__name__}: {exc}")

    try:
        chains = module.graph_structure_reasoning(sub, query)
    except Exception as exc:
        rep.check(False, "graph_structure_reasoning 不抛异常", f"{type(exc).__name__}: {exc}")
        return

    rep.check(isinstance(chains, list), "返回 list", type(chains).__name__)
    if not chains:
        rep.check(False, "生成推理链非空", "未生成任何推理链")
        return

    rep.check(all(isinstance(c, str) and c.strip() for c in chains), "推理链均为非空字符串", f"{len(chains)} 条")

    templated = [c for c in chains if c.strip().startswith("基于") and c.strip().endswith("的推理链")]
    rep.check(not templated, "推理链不是模板占位串", f"模板串={templated[:2]}")

    known_names = set(_node_names(sub.connected_nodes)) | set(_node_names(sub.central_nodes))
    grounded = [c for c in chains if any(n and n in c for n in known_names)]
    rep.check(bool(grounded), "推理链可溯源到真实节点名",
              f"{len(grounded)}/{len(chains)} 条命中真实节点")

    for c in chains[:3]:
        print(f"       · {c[:150]}")


def verify_graph_rag_search(module: GraphRAGRetrieval, rep: _Reporter) -> None:
    """
    验证「主搜索接口」：五类查询都必须能返回结构合法、可排序、可溯源的 Document。

    不变量：
      - 返回 List[Document]，条数 <= top_k
      - 每条 Document 的 page_content 非空，metadata 含 search_type
      - 结果按 relevance_score 降序（排序链路真实生效）
      - 非空结果的条数应 > 0（不能静默返回空，尤其 CLUSTERING 分支）
    """
    _section("能力6：主检索接口 graph_rag_search")

    queries = [
        ("实体关系", "宫保鸡丁和花生有什么关系？"),
        ("多跳", "鸡肉配什么蔬菜？"),
        ("子图", "川菜有什么特色菜"),
        ("路径", "从食材到成品菜的制作路径是怎样的？"),
        ("聚类", "和宫保鸡丁类似的菜有哪些？"),
    ]

    top_k = 5
    for label, q in queries:
        try:
            docs = module.graph_rag_search(q, top_k=top_k)
        except Exception as exc:
            rep.check(False, f"graph_rag_search({label}) 不抛异常", f"{type(exc).__name__}: {exc}")
            continue

        if not rep.check(isinstance(docs, list), f"返回 list（{label}）", type(docs).__name__):
            continue
        rep.check(len(docs) <= top_k, f"结果数不超过 top_k（{label}）", f"{len(docs)} <= {top_k}")

        if not docs:
            rep.check(False, f"返回非空结果（{label}）", "空结果，该查询分支疑似未接通")
            continue

        rep.check(all(getattr(d, "page_content", "").strip() for d in docs),
                  f"page_content 非空（{label}）")
        rep.check(all("search_type" in getattr(d, "metadata", {}) for d in docs),
                  f"metadata 含 search_type（{label}）",
                  f"{[d.metadata.get('search_type') for d in docs]}")

        scores = [float(d.metadata.get("relevance_score", 0.0)) for d in docs]
        rep.check(scores == sorted(scores, reverse=True),
                  f"结果按 relevance_score 降序（{label}）", f"{[round(s, 4) for s in scores]}")

        print(f"    -> {label}: {len(docs)} 条 | search_type={docs[0].metadata.get('search_type')} "
              f"| top-score={scores[0]:.4f}")
        print(f"       {docs[0].page_content[:130].replace(chr(10), ' ')}")


def verify_defensive_behavior(module: GraphRAGRetrieval, rep: _Reporter) -> None:
    """
    验证「降级与防御」：未建立 Neo4j 连接时，所有入口必须安全返回空结果而非抛异常。

    这是企业级可用性的底线：图库抖动不应该把整个 RAG 链路打挂。
    """
    _section("能力7：降级与防御（driver 未就绪）")

    offline = GraphRAGRetrieval(config=module.config, llm_client=module.llm_client)
    # 刻意不调用 initialize()，driver 保持 None
    offline.driver = None

    try:
        rep.check(offline.graph_rag_search("宫保鸡丁", top_k=3) == [], "graph_rag_search 安全返回空")
    except Exception as exc:
        rep.check(False, "graph_rag_search 安全返回空", f"{type(exc).__name__}: {exc}")

    for name, fn in (
        ("multi_hop_traversal", lambda: offline.multi_hop_traversal(
            GraphQuery(query_type=QueryType.MULTI_HOP, source_entities=["宫保鸡丁"]))),
        ("extract_knowledge_subgraph", lambda: offline.extract_knowledge_subgraph(
            GraphQuery(query_type=QueryType.SUBGRAPH, source_entities=["宫保鸡丁"]))),
    ):
        try:
            result = fn()
            rep.check(result is not None, f"{name} 安全降级不抛异常", type(result).__name__)
        except Exception as exc:
            rep.check(False, f"{name} 安全降级不抛异常", f"{type(exc).__name__}: {exc}")

    # 缓存与索引属性必须存在且类型正确（扩展性与可观测性的基础）
    rep.check(isinstance(getattr(module, "subgraph_cache", None), dict), "subgraph_cache 存在且为 dict")
    rep.check(isinstance(getattr(module, "entity_cache", None), dict), "entity_cache 存在且为 dict")
    rep.check(isinstance(getattr(module, "relation_cache", None), dict), "relation_cache 存在且为 dict")

    offline.close()


# ============================ 主流程 ============================


def main() -> int:
    """串联全部验证环节；任一环节的初始化失败都会直接给出可读的失败原因。"""
    print("=" * 64)
    print("GraphRAGRetrieval 检索能力验证")
    print("=" * 64)

    rep = _Reporter()

    config = GraphRAGConfig()
    llm_client = _build_llm_client()
    if llm_client is None:
        rep.warn("LLM 客户端", "未配置 OPENAI_API_KEY，意图理解将走降级路径，相关断言会放宽")

    module = GraphRAGRetrieval(config=config, llm_client=llm_client)

    print("\n[初始化] 连接 Neo4j 并构建图索引 ...")
    try:
        module.initialize()
    except Exception as exc:
        print(f"[致命] initialize() 抛出异常：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    if not getattr(module, "driver", None):
        print("[致命] Neo4j 未连接成功。请先执行 `docker compose up -d` 并等待 neo4j 健康。")
        return 1

    print(f"[初始化] 索引规模：实体缓存 {len(module.entity_cache)} 条，关系类型 {len(module.relation_cache)} 种")
    if not module.entity_cache:
        rep.warn("图索引", "entity_cache 为空，可能图谱未导入数据")

    # 依次验证七个能力点；任一环节内部异常都被捕获，不影响其余环节
    stages: List[Tuple[str, Callable[[GraphRAGRetrieval, _Reporter], None]]] = [
        ("查询意图理解", verify_intent_understanding),
        ("动态查询规划", verify_query_planning),
        ("多跳图遍历", verify_multi_hop_traversal),
        ("子图提取", verify_subgraph_extraction),
        ("图结构推理", verify_graph_reasoning),
        ("主检索接口", verify_graph_rag_search),
        ("降级与防御", verify_defensive_behavior),
    ]

    for name, fn in stages:
        try:
            fn(module, rep)
        except Exception as exc:
            rep.check(False, f"{name} 环节未崩溃", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    module.close()
    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(main())
