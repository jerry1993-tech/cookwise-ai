"""结果输出：路径 / 子图渲染为 Document、图结构分计算与四分量融合排序。"""

import logging
from collections import defaultdict
from typing import Dict, List

from langchain_core.documents import Document

from .shared_constants_helpers_models import (
    GraphPath,
    KnowledgeSubgraph,
    _DEFAULT_RELATION_WEIGHT,
    _FUSE_WEIGHT_LEXICAL,
    _FUSE_WEIGHT_PRIOR,
    _FUSE_WEIGHT_STRUCT,
    _FUSE_WEIGHT_TRACE,
    _LEXICAL_BIGRAM_WEIGHT,
    _RELATION_WEIGHTS,
    _SEARCH_TYPE_PRIOR,
    _char_bigrams,
    _char_unigrams,
    _containment,
    _dedup_preserve_order,
)

logger = logging.getLogger(__name__)


class DocumentFormattingMixin:
    """结果表达：把路径/子图渲染为自然语言描述与 Document，并计算图结构分。"""

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象。

        metadata 里同时写入：
          - 原有字段（search_type / path_length / relevance_score / path_type / node_count /
            relationship_count / recipe_name），保持与既有调用方兼容；
          - 新增字段（graph_struct_score / node_ids / path_text / retrieval_level / matched_query），
            分别用于最终排序融合、溯源、以及生成模块的层级标注（generation_integration 会读 retrieval_level）。
        """
        documents = []

        for path in paths:
            # 构建路径描述
            path_desc = self._build_path_description(path)

            # search_type 由路径类型映射而来，保证「实体关系 / 最短路径 / 聚类」都能被下游区分
            search_type_map = {
                "entity_relation": "graph_entity_relation",
                "shortest_path": "graph_path",
                "cluster_similar": "graph_cluster",
                "cluster_category": "graph_cluster",
                "cluster_cuisine": "graph_cluster",
            }
            search_type = search_type_map.get(path.path_type, "graph_path")

            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": search_type,
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "recipe_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果",
                    # --- 以下为新增字段 ---
                    "graph_struct_score": self._path_struct_score(path),
                    # node_ids / node_names 只保留【有真实 nodeId 的节点】：
                    # Category / ConceptType 这类节点的 nodeId 为 NULL，原样拼进来会得到
                    # 一批空串，既无法溯源反查，也与子图型 Document 的口径自相矛盾
                    # （子图侧已按同一规则过滤空值）。
                    # 【两个列表必须用同一掩码过滤】否则长度不同、下标不再一一对应，
                    # 下游任何 `zip(node_ids, node_names)` 都会错位配对。
                    # 被过滤掉的节点名仍完整保留在 path_text（路径描述文本）里，不丢信息。
                    "node_ids": [n.get("id") for n in path.nodes if n.get("id")],
                    "node_names": [str(n.get("name") or "") for n in path.nodes if n.get("id")],
                    "path_text": path_desc,
                    # 沿用项目既有的 LightRAG 层级约定（hybrid_retrieval.py 用 'entity' / 'topic'）：
                    # 路径型结果承载的是具体实体之间的关系，属于低层级(entity)。
                    # 若在此另造 'graph_path' 这类新值，未来任何按 retrieval_level 分组/筛选的编排代码
                    # 都会因为出现第三种取值而走进未预期的分支。细粒度的类型区分放在 graph_retrieval_level 里。
                    "retrieval_level": "entity",
                    "graph_retrieval_level": search_type,
                    "matched_query": query,
                }
            )
            documents.append(doc)

        return documents

    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph,
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为Document对象。

        除了子图整体描述，还会把每条推理链单独产出一份 Document：
        推理链是「已经被图结构验证过的结论」，作为独立的检索证据比塞在某个字段里更容易被生成模型引用。

        【空子图必须返回空列表，不能产出 Document】
        _fallback_subgraph_extraction 三级降级全失败时返回的是「空子图」而不是 None，
        若不判空就会生成一条
        「关于 该实体 的知识网络，包含 0 个相关概念和 0 个关系。」的 Document，
        且 relevance_score 仍有 0.4 —— 它会作为"检索证据"被 generation_integration
        拼成 [TOPIC] ... 直接塞进生成 prompt，等于凭空给模型递了一份内容为空的伪证据。
        更糟的是它让"图库全挂"这一最需要被发现的场景在测试里也表现为 PASS
        （断言"结果非空 / page_content 非空"都会被它骗过）。
        因此这里与 graph_rag_search 的降级分支保持同一判据：三个集合全空即视为无证据。

        降级可观测性：降级产生的子图会把 degraded / source 一并写进 metadata，
        让上层能区分「真的检索到了」与「图库不可用只能给个空壳」。
        """
        documents = []

        # 无任何节点与关系 => 「知识网络」这份描述没有任何可交付的内容，跳过它。
        # 注意这里【只跳过子图描述本身】，不提前 return：reasoning_chains 是独立证据，
        # 即便子图退化也不能连带丢弃（虽然实践中空子图不会产出推理链，但不做这个假设更稳）。
        has_content = bool(subgraph.central_nodes or subgraph.connected_nodes or subgraph.relationships)
        if not has_content:
            logger.warning("子图为空（通常意味着图库不可用或实体不在图中），跳过子图描述文档")
        else:
            # 子图整体描述
            subgraph_desc = self._build_subgraph_description(subgraph)

            doc = Document(
                page_content=subgraph_desc,
                metadata={
                    "search_type": "knowledge_subgraph",
                    "node_count": len(subgraph.connected_nodes),
                    "relationship_count": len(subgraph.relationships),
                    "graph_density": subgraph.graph_metrics.get("density", 0.0),
                    "reasoning_chains": reasoning_chains,
                    "recipe_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图",
                    # --- 以下为新增字段 ---
                    "graph_struct_score": self._subgraph_struct_score(subgraph),
                    # 子图的"相关性"取自身图结构与推理链数量：推理链越多说明该子图对问题越有解释力
                    "relevance_score": round(min(0.4 + 0.1 * len(reasoning_chains), 0.9), 6),
                    # 去重：同一个节点可能既是某个种子的中心节点，又是另一个种子的一跳邻居，
                    # 直接拼接会得到「49 个节点却有 54 个 node_id」这种自相矛盾的元数据，影响下游溯源与统计。
                    # node_ids 只保留【真实的 nodeId】：Category / ConceptType 这类节点的
                    # nodeId 为 NULL，若原样拼进来会得到一批空串，既无法用于溯源反查，
                    # 也会让下游"N 个节点却有 M 个 node_id"的统计失真。
                    # 这些节点的名称仍完整保留在子图描述文本与 node_names 中，不影响可读性与引用。
                    "node_ids": _dedup_preserve_order([
                        n.get("id", "") for n in (subgraph.central_nodes + subgraph.connected_nodes)
                        if n.get("id")
                    ]),
                    # 子图承载的是主题级知识网络，按既有约定属于高层级(topic)
                    "retrieval_level": "topic",
                    "graph_retrieval_level": "knowledge_subgraph",
                    "matched_query": query,
                    # 降级可观测性：_fallback_subgraph_extraction 会在 graph_metrics 里写
                    # degraded / source 两个标记，这里把它透传到 Document 上。
                    # 不这么做的话，降级状态只存在于子图对象内部，随 Document 一起被丢弃，
                    # 上层（生成模块 / 监控 / 测试断言）完全看不出"这条结果是图库挂了的产物"。
                    "degraded": bool(subgraph.graph_metrics.get("degraded")),
                    "degraded_source": subgraph.graph_metrics.get("source", ""),
                }
            )
            documents.append(doc)

        # 推理链各自的 Document：便于下游按条引用，也便于排序时按证据强度区分
        for index, chain in enumerate(reasoning_chains or []):
            if not chain or not str(chain).strip():
                continue
            documents.append(Document(
                page_content=str(chain),
                metadata={
                    "search_type": "graph_reasoning_chain",
                    "chain_index": index,
                    "reasoning_chains": [str(chain)],
                    "recipe_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图",
                    "graph_struct_score": self._subgraph_struct_score(subgraph),
                    # 推理链的初始相关性随其在 _validate_reasoning_chains 中的排序位置衰减
                    "relevance_score": round(max(0.5 - 0.05 * index, 0.2), 6),
                    # 推理链是主题级的结论，同样归入高层级(topic)
                    "retrieval_level": "topic",
                    "graph_retrieval_level": "graph_reasoning_chain",
                    "matched_query": query,
                }
            ))

        return documents

    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述。

        输出形如：宫保鸡丁 --[REQUIRES]--> 鸡胸脯肉 --[BELONGS_TO_CATEGORY]--> 蛋白质
        要点：
          - 关系方向依据 _rel_to_dict 记录的首尾 nodeId 判定（--> 或 <--），不再无脑画成单向；
          - 关系与节点的下标严格对齐（第 i 条边连接第 i、i+1 个节点）；
          - REQUIRES 这类带量属性的边会附带 amount/unit，让描述可直接用于回答"放多少"。
        """
        if not path.nodes:
            return "空路径"

        names = [str(node.get("name", f"节点{i}")) for i, node in enumerate(path.nodes)]

        # 完全没有关系边的路径（例如「同菜系」这类属性级同组——它不靠图上的边连通）
        # 不能走下面的箭头渲染，否则节点名会被直接拼接成「宫保鸡丁水煮鱼」这种没有分隔的乱串。
        if not path.relationships:
            return (f"同一分组内的相关条目：{'、'.join(names)}"
                    f"\n（共 {len(names)} 个条目；它们之间没有直接的图关系边，"
                    f"依据的是属性同组，如同菜系/同分类）")

        desc_parts = []
        for i, node in enumerate(path.nodes):
            desc_parts.append(names[i])
            if i < len(path.relationships):
                rel = path.relationships[i]
                rel_type = str(rel.get("type") or "相关")
                # 判断方向：当前节点是这条边的起点则画 -->，否则画 <--
                current_id = str(node.get("id") or "")
                start_id = str(rel.get("start_id") or "")
                arrow = "-->" if (not start_id or current_id == start_id) else "<--"

                detail = ""
                properties = rel.get("properties") or {}
                amount, unit = properties.get("amount"), properties.get("unit")
                if amount and unit:
                    detail = f"({amount}{unit})"

                desc_parts.append(f" -[{rel_type}]{detail}{arrow} ")
            elif i != len(path.nodes) - 1:
                # 关系条数少于节点间隔时的兜底分隔，保证任何情况下都不会出现两个名字粘连
                desc_parts.append(" —— ")

        description = "".join(desc_parts)

        # 追加一行结构摘要，帮助生成模型快速抓住路径的规模与可信度
        description += (f"\n（路径长度：{path.path_length} 跳；"
                        f"涉及 {len(path.nodes)} 个节点、{len(path.relationships)} 条关系；"
                        f"关系类型：{'、'.join([str(r.get('type') or '') for r in path.relationships]) or '无'}）")

        # 搭配共现路径额外给出「共现强度」这一行：
        # 「鸡肉 -[REQUIRES]-> 麻辣香锅 <-[REQUIRES]- 青椒」这条结构本身不含"有多常见"的信息，
        # 而共现菜谱数正是这条搭配是否可信的唯一依据。把它渲染进正文，
        # 生成模型才可能说出"在 5 个菜谱里鸡肉和青椒一起出现"这类可核查的表述，
        # 而不是把一次偶然的共现说成像普遍规律。
        support, src_count = self._pairing_support(path)
        if support:
            description += (f"\n（搭配共现证据：在 {support} 个菜谱中同时出现"
                            f"{f'，覆盖 {src_count} 种同族食材' if src_count > 1 else ''}）")
        return description

    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述。

        以「中心实体 → 属性画像 → 组成食材（按类别） → 制作步骤 → 关联条目 → 图指标」的顺序组织，
        让这段文本本身就能回答相当一部分问题，而不必依赖生成模型二次归纳。
        """
        if subgraph is None:
            return "知识子图不可用。"

        central_names = [node.get("name", "未知") for node in (subgraph.central_nodes or [])]
        node_count = len(subgraph.connected_nodes or [])
        rel_count = len(subgraph.relationships or [])

        parts: List[str] = [
            f"关于 {', '.join([str(n) for n in central_names]) or '该实体'} 的知识网络，"
            f"包含 {node_count} 个相关概念和 {rel_count} 个关系。"
        ]

        # 1) 中心实体画像：把菜系/分类/难度这类"一句话画像"信息前置，因为它们最常被问到
        for node in (subgraph.central_nodes or []):
            properties = node.get("properties") or {}
            profile_bits = []
            for key, label in (("category", "分类"), ("cuisineType", "菜系"),
                               ("difficulty", "难度"), ("cookTime", "烹饪时间"),
                               ("prepTime", "准备时间"), ("servings", "份量")):
                value = properties.get(key)
                if value not in (None, "", []):
                    profile_bits.append(f"{label}: {value}")
            if properties.get("description"):
                profile_bits.append(f"简介: {str(properties['description'])[:120]}")
            if profile_bits:
                parts.append(f"「{node.get('name', '')}」的基本信息 —— " + "；".join(profile_bits) + "。")

        # 2) 食材：按 category 分组，这是烹饪问答里最核心的信息
        ingredients = [n for n in (subgraph.connected_nodes or []) if "Ingredient" in (n.get("labels") or [])]
        if ingredients:
            by_category: Dict[str, List[str]] = defaultdict(list)
            for item in ingredients:
                category = str((item.get("properties") or {}).get("category") or "未分类")
                by_category[category].append(str(item.get("name") or ""))
            chunks = [
                f"{category}（{'、'.join([n for n in names if n][:8])}）"
                for category, names in sorted(by_category.items(), key=lambda kv: -len(kv[1]))
            ]
            parts.append(f"涉及食材共 {len(ingredients)} 种，按类别：" + "；".join(chunks) + "。")

        # 3) 步骤
        steps = [n for n in (subgraph.connected_nodes or []) if "CookingStep" in (n.get("labels") or [])]
        if steps:
            parts.append(f"包含 {len(steps)} 个制作步骤。")

        # 4) 关联条目：分类/相似菜谱/难度层级等
        related_recipes = [str(n.get("name") or "") for n in (subgraph.connected_nodes or [])
                           if "Recipe" in (n.get("labels") or []) and n.get("name")]
        if related_recipes:
            parts.append(f"关联菜谱：{'、'.join(related_recipes[:8])}。")

        categories = [str(n.get("name") or "") for n in (subgraph.connected_nodes or [])
                      if "Category" in (n.get("labels") or []) and n.get("name")]
        if categories:
            parts.append(f"所属分类：{'、'.join(categories[:8])}。")

        # 5) 图指标：给出量化特征，便于上层判断该子图的证据强度
        metrics = subgraph.graph_metrics or {}
        if metrics:
            parts.append(
                f"图指标：节点 {int(metrics.get('node_count', 0))}、关系 {int(metrics.get('relationship_count', 0))}、"
                f"密度 {metrics.get('density', 0)}、平均度数 {metrics.get('avg_degree', 0)}。"
            )

        return "\n".join(parts)

    def _path_struct_score(self, path: GraphPath) -> float:
        """路径的图结构分（不含词法证据），取值 [0,1]。

        由「平均关系权重」「路径紧凑度」，以及「搭配共现强度」（仅搭配路径有）三者构成：
        语义关系越强、跳数越少、该搭配被越多菜谱支持，结构证据越硬。
        """
        if not path.relationships:
            rel_strength = _DEFAULT_RELATION_WEIGHT
        else:
            rel_strength = sum(
                _RELATION_WEIGHTS.get(str(r.get("type") or ""), _DEFAULT_RELATION_WEIGHT)
                for r in path.relationships
            ) / float(len(path.relationships))
        compactness = 1.0 / (1.0 + max(int(path.path_length or 1), 1))
        # 搭配共现强度计入结构分：这条路走的是【融合排序】的 struct 分量
        # （_rank_by_graph_relevance 的 0.25 权重），只把加成写进 path.relevance_score
        # 是不够的 —— 那个字段在融合阶段会被 final_score 覆盖，属于"算了但没人用"。
        return round(min(0.7 * rel_strength + 0.3 * compactness + self._pairing_support_bonus(path), 1.0), 6)

    def _subgraph_struct_score(self, subgraph: KnowledgeSubgraph) -> float:
        """子图的图结构分（不含词法证据），取值 [0,1]。

        由「节点规模」「关系规模」「密度」三者加权：规模说明覆盖面，密度说明内部关联紧密度。
        规模分母取 60（实测一个菜谱的完整 1 跳邻域约在 30~60 个节点量级），密度乘以 20 后饱和。
        """
        if subgraph is None:
            return 0.0
        node_count = len(subgraph.connected_nodes or []) + len(subgraph.central_nodes or [])
        rel_count = len(subgraph.relationships or [])
        density = float((subgraph.graph_metrics or {}).get("density", 0.0) or 0.0)

        coverage = min(node_count / 60.0, 1.0)
        linkage = min(rel_count / 60.0, 1.0)
        tightness = min(density * 20.0, 1.0)
        return round(min(0.5 * coverage + 0.3 * linkage + 0.2 * tightness, 1.0), 6)

class ResultRankingMixin:
    """融合排序：词法重合度 + 图结构分 + 结果类型先验 + 可溯源程度的四分量排序。"""

    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序（四分量融合排序）。

        融合公式（权重见模块常量）：
            final = 0.40 * 词法重合度
                  + 0.25 * 图结构分（路径强度 / 子图覆盖率，由 _path_struct_score / _subgraph_struct_score 预计算）
                  + 0.20 * 结果类型先验（知识子图 > 实体关系 > 路径 > 聚类）
                  + 0.15 * 可溯源程度（是否带 node_id / reasoning_chains / 节点列表等可核查线索）

        这么设计的原因：单看词法会把「提到很多食材的长子图」排到前面，单看图结构又会忽视问题本身，
        四分量各自负责一类证据，且权重可通过模块常量按业务反馈调整。

        副作用（有意为之）：把融合分写回 metadata["relevance_score"]，
        这样「按 relevance_score 降序」在下游始终成立，调用方无需知道融合公式的存在。
        """
        if not documents:
            return []

        query_bigrams = _char_bigrams(query or "")
        query_unigrams = _char_unigrams(query or "")

        for doc in documents:
            metadata = doc.metadata or {}
            content = doc.page_content or ""

            # 1) 词法重合度：bigram 体现短语级匹配，unigram 兜底短查询
            bigram_score = _containment(query_bigrams, _char_bigrams(content)) if query_bigrams else 0.0
            unigram_score = _containment(query_unigrams, _char_unigrams(content)) if query_unigrams else 0.0
            lexical = (_LEXICAL_BIGRAM_WEIGHT * bigram_score
                       + (1.0 - _LEXICAL_BIGRAM_WEIGHT) * unigram_score)

            # 2) 图结构分：优先取预计算结果；没有则退回按已有指标现算
            struct = metadata.get("graph_struct_score")
            if not isinstance(struct, (int, float)):
                struct = 0.0

            # 3) 结果类型先验
            prior = _SEARCH_TYPE_PRIOR.get(str(metadata.get("search_type") or ""), 0.6)

            # 4) 可溯源程度：能指回具体图谱线索的结果更值得采信
            trace = 0.0
            if metadata.get("node_ids") or metadata.get("node_id"):
                trace += 0.4
            if metadata.get("recipe_name"):
                trace += 0.2
            if metadata.get("reasoning_chains"):
                trace += 0.2
            if metadata.get("node_names") or metadata.get("path_text"):
                trace += 0.2
            # 推理链之间：保留 _validate_reasoning_chains 已判定的优劣次序（chain_index 越小越好）。
            # 不加这一项的话，同一子图产出的多条推理链会拿到完全相同的融合分，排序退化为「依赖输入顺序」。
            chain_index = metadata.get("chain_index")
            if isinstance(chain_index, int) and chain_index >= 0:
                trace += max(0.2 - 0.05 * chain_index, 0.0)

            final_score = (
                _FUSE_WEIGHT_LEXICAL * lexical
                + _FUSE_WEIGHT_STRUCT * float(struct)
                + _FUSE_WEIGHT_PRIOR * float(prior)
                + _FUSE_WEIGHT_TRACE * min(trace, 1.0)
            )

            # 保留原始图分以供排查，同时把融合分写回 relevance_score 保证排序口径统一
            metadata.setdefault("graph_relevance_score", metadata.get("relevance_score", 0.0))
            metadata["lexical_score"] = round(float(lexical), 6)
            metadata["graph_struct_score"] = round(float(struct), 6)
            metadata["final_score"] = round(float(final_score), 6)
            metadata["relevance_score"] = round(float(final_score), 6)

        return sorted(documents,
                     key=lambda x: x.metadata.get("relevance_score", 0.0),
                     reverse=True)
