"""实体名索引与解析：名称索引预热、实体名规范化、别名扩展、候选实体抽取与种子节点解析。"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .shared_constants_helpers_models import (
    _ALIAS_MAX_NAMES,
    _ALIAS_MAX_NODE_IDS,
    _ALIAS_SPARSITY_THRESHOLD,
    _EDIBLE_PART_MORPHEMES,
    _GENERIC_MATERIAL_MORPHEMES,
    _GENERIC_TARGET_KEYWORDS,
    _GRAPH_INDEX_LIMIT,
    _PLACEHOLDER_NAMES,
    _PLACEHOLDER_NODE_IDS,
    _SPECIES_MORPHEMES,
    _TARGET_SUFFIX_NOISE,
    _dedup_preserve_order,
)

logger = logging.getLogger(__name__)


class GraphIndexMixin:
    """实体词表与种子解析：预热图索引、规范化实体名、同族别名对齐、候选实体抽取与种子节点解析。"""

    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")

        try:
            with self.driver.session() as session:
                # 构建实体索引 - 修复Neo4j语法兼容性问题
                # 说明：相比早期版本增加了标签白名单与占位节点过滤，避免把 GraphStatistics 等统计节点索引进来；
                # LIMIT 从 1000 提升到 _GRAPH_INDEX_LIMIT，让实体解析覆盖到绝大多数业务节点。
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                  AND NOT coalesce(n.nodeId, '') IN $placeholder_ids
                  AND (n:Recipe OR n:Ingredient OR n:CookingStep OR n:Category OR n:ConceptType OR n:DifficultyLevel)
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id,
                       n.name as name, n.category as category,
                       // 菜系也一并取回：它是"只作为属性存在"的实体词（如「川菜」），
                       // 多取一列的代价为 0（同一次全表扫描），却让本地兜底的实体抽取覆盖到这类词。
                       n.cuisineType as cuisine_type, degree
                ORDER BY degree DESC, node_id
                LIMIT $limit
                """

                result = session.run(entity_query, {
                    "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                    "limit": _GRAPH_INDEX_LIMIT,
                })
                self.entity_cache.clear()
                self._name_index.clear()
                self._vocab_phrases.clear()
                for record in result:
                    node_id = record["node_id"]
                    name = record["name"]
                    if not node_id:
                        continue
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": name,
                        "category": record["category"],
                        "degree": record["degree"]
                    }
                    # 同步维护 名称 -> nodeId 反向索引，供 _candidate_entities_from_query 做 O(1) 命中
                    if name and str(name) not in _PLACEHOLDER_NAMES:
                        self._name_index[str(name)].append(str(node_id))
                    # 属性值词表：分类与菜系按【短语】收录，并先按分隔符拆开。
                    # 拆分的必要性：图谱里存在「陕菜,豫菜,浙菜,鲁菜,川菜,淮扬菜」这类
                    # 多值属性（一条记录里塞了多个菜系），整串收进来时「川菜」既是它自己的
                    # 精确命中、又是这个复合串的子串，两个候选会一起占掉候选名额。
                    # 只收录长度 >= 2 的短语：单字值（如分类里的「菜」）作为查询子串会命中几乎所有问句。
                    for raw_phrase in (record["category"], record["cuisine_type"]):
                        for phrase in re.split(r"[,，、/|;；]+", str(raw_phrase or "")):
                            phrase = phrase.strip()
                            if len(phrase) >= 2 and phrase not in _PLACEHOLDER_NAMES:
                                self._vocab_phrases.add(phrase)

                # 构建关系类型索引
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """

                result = session.run(relation_query)
                self.relation_cache.clear()
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]

                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def _normalize_entity_name(self, name: Any) -> str:
        """规范化实体名：去掉首尾空白、引号与括号、多余标点。

        来源可能是 LLM 输出（常带回引号、书名号）或用户输入，统一清洗后才能真正对上线上的节点名。
        """
        if name is None:
            return ""
        text = str(name).strip()
        text = re.sub(r'^[\s"\'「」『』《》【】\[\]（）()]+', "", text)
        text = re.sub(r'[\s"\'「」『』《》【】\[\]（）()]+$', "", text)
        return text.strip()

    def _normalize_target_keywords(self, targets: Optional[List[str]]) -> List[str]:
        """把「目标实体」描述规范成可用于匹配的关键词。

        understand_graph_query 产出的 target_entities 常常是【类型描述】（如「蔬菜类食材」）而不是图谱里的真实名字，
        直接拿去和 labels(target) 比对必然落空。这里剥掉「类食材 / 类别 / 类」这类后缀噪声，
        让「蔬菜类食材」能退化成「蔬菜」，从而对上 Category 节点与 Ingredient.category 属性。
        """
        keywords: List[str] = []
        for raw in (targets or []):
            text = self._normalize_entity_name(raw).lower()
            if not text:
                continue

            # 先尝试剥离「类食材 / 类别 / 类」这类后缀噪声：
            # 剥离成功说明原始串只是个"类型描述"（如「蔬菜类食材」），此时只保留剥离后的具体词（「蔬菜」），
            # 不能把描述串本身也当关键词，否则它永远匹配不到任何节点名。
            trimmed = ""
            for suffix in _TARGET_SUFFIX_NOISE:
                if text.endswith(suffix) and len(text) > len(suffix):
                    candidate = text[: -len(suffix)].strip()
                    if candidate:
                        trimmed = candidate
                    break

            # 元类型词不参与过滤：丢弃它等于「不加约束」，比强行过滤把结果筛空要好得多
            for candidate in ([trimmed] if trimmed else [text]):
                if candidate and candidate not in _GENERIC_TARGET_KEYWORDS:
                    keywords.append(candidate)

        return _dedup_preserve_order(keywords)

    def _iter_ingredient_name_categories(self) -> Dict[str, Set[str]]:
        """汇总「食材名 -> 该类目集合」的词表视图（只读内存索引，不查库）。

        为什么单独抽一层：查询词与图谱标准词的比对需要一个"图谱真实词表"，
        而它已经在 initialize() 时预热成了 _name_index / entity_cache。
        这里把它整理成名字级的视图，好处是
          （1）零额外 Cypher 往返 —— 对齐判定完全在内存里做，实测 0.1ms 量级；
          （2）只收 :Ingredient —— 图谱里「宫保鸡丁」「可乐鸡翅」这类【菜名】与食材同名不同义，
               若一起参与对齐，"鸡肉"会扩展到菜名，把检索的语义粒度改坏（食材级 -> 菜谱级）。

        Returns:
            {食材名: {该类目名, ...}}；名字为空或占位名不收录。
        """
        name_categories: Dict[str, Set[str]] = {}
        for info in self.entity_cache.values():
            if "Ingredient" not in list(info.get("labels") or []):
                continue
            name = str(info.get("name") or "")
            if not name or name in _PLACEHOLDER_NAMES:
                continue
            name_categories.setdefault(name, set()).add(str(info.get("category") or ""))
        return name_categories

    def _expand_alias_names(self, names: List[str]) -> List[str]:
        """把查询词对齐到图谱真实词表，返回【同族食材名】（纯内存，不查库）。

        用途：解决"用户/LLM 用词与图谱标准词错位"导致的召回空转。
        典型场景：图谱里鸡肉类食材叫「鸡腿肉 / 鸡胸肉 / 鸡翅中 / …」，
        而查询说「鸡肉」，子串匹配只能命中 2 个名字 —— 对齐后能覆盖 12 个名字、
        27 种共现蔬菜（实测）。

        判定规则（三条通用信号，详见模块顶部 _EDIBLE_PART_MORPHEMES 注释）：
            1. 稀疏触发：精确/子串命中名字数 < _ALIAS_SPARSITY_THRESHOLD 才扩展，
               命中充足时原样返回，避免用同族名去稀释本来准确的种子；
            2. 末字必须是可食用部位/切割形态语素（挡住「鸡蛋」「鸡汤」「面粉」这类词）；
            3. 头语素 + 同类目：候选名要包含头语素，且与参照节点同 category。

        设计取舍：
          * 不做单向的"硬编码同义词表" —— 词表随图谱演进而失效，且无法复用到别的食材；
            这里词表来自图谱自身（_name_index），规则只描述"什么算同族"，因此
            换一份菜谱图谱（或换成服装/医药图谱）时只需替换语素集合，逻辑不动。
          * 不猜、不生成 —— 只做"图谱里已存在的名字"之间的收敛，绝不产生图谱里没有的实体名，
            保证后续所有查询都能落到真实节点上。

        Args:
            names: 已规范化的查询词列表（来自 _resolve_seed_nodes 的实体名）

        Returns:
            同族别名列表（已去重、已按 _ALIAS_MAX_NAMES 截断）；无需扩展时返回 []
        """
        if not names:
            return []

        vocabulary = self._iter_ingredient_name_categories()
        if not vocabulary:
            # 索引未预热（例如降级路径）：不做任何猜测，交回原链路
            return []

        aliases: List[str] = []
        for raw in names:
            query = self._normalize_entity_name(raw)
            if len(query) < 2:
                continue

            # 规则 2：末字必须是可食用部位/切割形态语素。
            # 「鸡蛋」「土豆」「面粉」「鸡蛋清」在这里就被挡住 —— 它们的"同族"是另一个具体食材，
            # 一旦扩展必然产生误召回（实测「鸡蛋」用 1 字头扩展会混进「鸡腿肉」）。
            if query[-1] not in _EDIBLE_PART_MORPHEMES:
                continue

            # 规则 1：稀疏触发。命中充足时不做扩展。
            matched = {name for name in vocabulary if name == query or query in name}
            if len(matched) >= _ALIAS_SPARSITY_THRESHOLD:
                continue

            # 参照类目：以精确命中优先，其次子串命中，取它们的类目并集作为"同族"的判据
            ref_categories: Set[str] = set()
            ordered_refs = ([query] if query in vocabulary else []) + sorted(matched)
            for ref in ordered_refs:
                ref_categories |= vocabulary.get(ref, set())
            ref_categories.discard("")

            def _family(head: str) -> List[str]:
                """按给定头语素收集同族名字（规则 3：头语素 + 末字部位语素 + 同类目）。

                额外一条【换物种否决】（见下方注释）：头语素之后紧跟另一个物种语素时丢弃该候选。
                """
                found: List[str] = []
                for candidate in sorted(vocabulary):
                    if len(candidate) < 2 or candidate == query or candidate in matched:
                        continue
                    if head not in candidate:
                        continue
                    if candidate[-1] not in _EDIBLE_PART_MORPHEMES:
                        continue
                    # 【换物种否决】以头语素开头的候选，若紧接着的第二个字又是物种语素，
                    # 说明它是"另一种动物 + 部位"（「牛蛙肉块」= 牛蛙 + 肉块），不是同族。
                    # 实测这是本规则唯一确凿的误召回：不加此条时「牛肉」的 9 个别名里混进
                    # 「牛蛙肉块」（牛蛙不是牛）；加了之后「牛肉」→8 个别名且全部为真。
                    # 只在 candidate 以 head 开头时检查：头语素出现在词中/词尾的候选
                    # （「大虾」「农村玉米鸡」「肥牛」）不存在这种"第二个物种"结构。
                    # 下一个字等于 head 本身不算换物种（「肥牛」的 head=牛、candidate[1]=牛）。
                    if (candidate.startswith(head) and len(candidate) > len(head)
                            and candidate[len(head)] in _SPECIES_MORPHEMES
                            and candidate[len(head)] != head):
                        continue
                    if ref_categories and not (vocabulary.get(candidate, set()) & ref_categories):
                        continue
                    found.append(candidate)
                return found

            # 主头语素：末字是"材料通称"时取主料语素（「鸡肉」->「鸡」），否则用完整词
            # （「鸡蛋」->「鸡蛋」、「虾仁」->「虾仁」、「鸡腿」->「鸡腿」）。
            primary = query[:-1] if query[-1] in _GENERIC_MATERIAL_MORPHEMES else query
            found = _family(primary)

            # 兜底头语素：主头语素一个字都没扩展到（例如「虾仁」在图谱里只有它自己）时，
            # 退回查询词的【首字】再试一次（「虾仁」->「虾」，可扩展到 8 个虾类食材名）。
            #
            # 【首字必须是物种语素，这一条不能省】只在"主头语素零产出"时放宽还不够：
            # 「排骨」的完整词同样零产出（能包含「排骨」的名字只有它自己和「猪排骨或五花肉」，
            # 都属于已命中集合），于是会退到「排」——把「羊排/牛排/带皮羊排」全算成同族，
            # 而「猪排骨」与「羊排」是两种不同的肉。实测「排骨」在无此限制时展开出
            # ['大排','带皮羊排','牛排','羊排','肋排']。
            # 「排」是切割形态语素而非物种语素，本就不该被当作"同族"的锚点；
            # 而「虾仁」的「虾」既是物种也是该词的主料，放宽才是对的。
            if not found and primary != query[:1] and query[0] in _SPECIES_MORPHEMES:
                found = _family(query[:1])

            aliases.extend(found)

        expanded = _dedup_preserve_order(aliases)[:_ALIAS_MAX_NAMES]
        if expanded:
            logger.info(f"实体词表对齐：由 {names} 扩展出 {len(expanded)} 个同族食材名 {expanded}")
        return expanded

    def _candidate_entities_from_query(self, query: str, limit: int = 5) -> List[str]:
        """从自然语言查询中抽取候选实体名（不依赖 LLM 的本地兜底）。

        为什么要它：意图理解的降级路径过去直接把整句查询当成实体名（source_entities=[query]），
        而图中节点名是「宫保鸡丁」这种短语，整句查询用 CONTAINS 永远匹配不到任何节点，等于整条检索链空转。
        这里改为拿查询去和已预热的词表做「完全相等 / 互为子串」比对，产出真正可落到图谱上的实体名。

        【两张词表都要参与比对】
          - _name_index：节点名（「宫保鸡丁」「鸡胸肉」…）；
          - _vocab_phrases：只作为【属性值】存在的词（菜系「川菜」、分类「蔬菜」…）。
        只看节点名的代价是实测过的：「川菜有什么特色菜」在只查节点名时零命中，
        函数会退回"把整句查询当实体名"，于是 _resolve_seed_nodes 拿到整句、匹配不到任何节点、
        整条检索链返回 0 条 —— 而同一个「川菜」若单独成词却能通过 Recipe.cuisineType 解析出 5 个种子。

        匹配打分：完全相等 1.0 > 词是查询子串（越长越高，最高 1.0）> 查询是词子串 0.6；
        同分时【节点名优先于属性值】（属性值是"类型"级别的词，节点名是"具体对象"，后者更可能正是用户所指）。

        Returns:
            候选实体名列表；词表完全没命中时退回 [整句查询]（让 _resolve_seed_nodes 的 CONTAINS 分支
            还有机会命中长节点名，例如节点名「红烧肉」比查询里的「红烧肉怎么做」更短的情形）。
        """
        normalized = self._normalize_entity_name(query)
        if not normalized:
            # 只有「非空字符串」才退回"把整句查询当实体名"的兜底行为；
            # None / 数字 / 列表等无意义输入一律返回空列表。
            # 【为什么】早期无条件 `return [query]`，在 query=None 时会产出
            # source_entities=[None] —— 这违反 GraphQuery.source_entities: List[str] 的契约，
            # 并让 _subgraph_cache_key 的 `",".join(sorted(...))` 抛 TypeError，
            # 异常被 graph_rag_search 的兜底 except 吞成"0 条结果"，
            # 把"调用方传错了类型"伪装成"图里没这个实体"。
            text = query if isinstance(query, str) else ""
            return [text] if text.strip() else []

        # (得分, 是否节点名, 名字长度, 词) —— 第 2 项用于"同分时节点名优先"
        scored: List[Tuple[float, int, int, str]] = []

        def _score_phrase(phrase: str, is_node_name: int) -> None:
            """给一个词表项打分并登记（规则见 docstring；长度 < 2 的词不参与子串匹配）。"""
            if not phrase or phrase in _PLACEHOLDER_NAMES:
                return
            if phrase == normalized:
                scored.append((1.0, is_node_name, len(phrase), phrase))
                return
            if len(phrase) < 2:
                return
            if phrase in normalized:
                # 词是查询的子串：越长的词越具体，得分越高
                ratio = min(len(phrase) / float(max(len(normalized), 1)), 1.0)
                scored.append((0.8 + 0.2 * ratio, is_node_name, len(phrase), phrase))
            elif normalized in phrase:
                scored.append((0.6, is_node_name, len(phrase), phrase))

        for name in list(self._name_index.keys()):
            _score_phrase(name, 1)
        for phrase in self._vocab_phrases:
            if phrase not in self._name_index:   # 已是节点名的词不重复登记
                _score_phrase(phrase, 0)

        # 先按得分降序，再按"节点名优先"，最后按长度降序（更长的词更具体）
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        candidates = _dedup_preserve_order([item[3] for item in scored])[:limit]
        return candidates or [normalized]

    def _resolve_seed_nodes(self, entities: Optional[List[str]], limit: int = 5,
                            session=None) -> List[Dict[str, Any]]:
        """把实体名解析成图谱中的真实种子节点。

        采用「多字段逐级放宽」的匹配策略（顺序即优先级）：
            1. 节点名精确相等
            2. 节点名包含实体名（处理「蔬菜（土豆片/豆芽…）」这类复合命名的食材节点）
            3. 菜系 / 分类 / 标签 / 描述等属性命中（关键：实测「川菜」在图谱里【不存在】同名节点，
               只存在于 Recipe.cuisineType，没有这一步「川菜有什么特色菜」整条链路会直接空转）
        每一级都强制标签白名单 + 占位节点过滤，并按度数降序取 top N，让信息量更大的节点优先成为种子。

        【第 2.5 级：同族别名（_expand_alias_names）】
        用户词与图谱词表对不上是实测出来的首要失败原因：图谱里只有 2 个节点的名字含「鸡肉」，
        而真实的鸡肉类食材名有 12 个（鸡胸肉/鸡翅中/鸡腿肉/鸡爪…）。
        因此解析前先把查询词扩展成同族食材名，并【按 nodeId 精确命中】（见下方 alias_names 注释），
        统一记为命中层级 2 —— 与"名字包含"同级，既参与了"名称级证据优先"的收口，又不会压过原词。

        【收口规则：名称级证据优先于属性级证据】
        候选行里只要存在名称级命中（层级 >= 2），就整体丢弃属性级命中（层级 1）。
        实测「鸡肉」的属性级命中是名为「意式烤鸡」的菜谱（度数 16）与若干步骤节点（度数 8~11），
        它们会把仅有 5 个名额的种子槽位挤满，而真正要遍历的食材节点（度数恒为 3）一个都进不来。

        【重要数据事实：食材节点按菜谱反规范化复制，同名节点大量存在】
        实测【鸡蛋】在库里有 72 个独立的 :Ingredient 节点（nodeId 各不相同），【洋葱】有 24 个，
        它们的度数几乎都是 3（1×REQUIRES + 1×BELONGS_TO_CATEGORY + 1×HAS_CONCEPT_TYPE）。
        因此对食材名做解析时：
          - degree 排序在这些同名节点之间【完全无法区分】（全等于 3），
            若只按 degree 排序，"取哪 5 个"由存储层决定 —— 同一个查询两次运行可能给出不同结果。
            为此排序追加了 coalesce(n.nodeId,'') 作为稳定兜底键，保证结果可复现
            （检索系统的可复现性比"多拿几个节点"更重要，否则缓存、评测、排障全都对不上账）；
          - 覆盖度天然是不完整的：limit 个同名节点只能覆盖到一部分菜谱。
            这是图谱建模导致的固有限制，不是解析缺陷，故此处只做确定性收口，不做跨节点聚合
            （聚合会改变结果的语义粒度，属于建模层决策，超出检索模块的职责）。

        Args:
            entities: 待解析的实体名列表（来自 LLM 或本地兜底）
            limit: 返回的种子节点数上限
            session: 可选的既有会话；传入则复用，避免重复建连

        Returns:
            节点字典列表（已带 degree 字段）
        """
        names = _dedup_preserve_order([
            self._normalize_entity_name(e) for e in (entities or []) if self._normalize_entity_name(e)
        ])
        if not names:
            return []

        # 【实体词表对齐】把查询词扩展成图谱真实词表里的同族食材名（纯内存判定，零额外往返）。
        # 这一步是"鸡肉配什么蔬菜"从 2 条结果变成几十条结果的关键：
        # 未对齐时『鸡肉』只能命中 2 个名字，对齐后覆盖 12 个鸡肉类食材名。
        # 别名与用户原词合并进同一条 Cypher 的 CALL{...UNION...} 里（各自一个分支），
        # 因此标签白名单、占位过滤、确定性排序全部复用，不存在第二条解析通路。
        lowered_names = {n.lower() for n in names}
        alias_names = [
            a for a in self._expand_alias_names(names) if a.lower() not in lowered_names
        ]
        # 别名直接用【内存索引里的 nodeId】定位，而不是拿别名再去扫名字：
        #   (1) 图谱为每个标签建了 nodeId 唯一索引（实测 ingredient_id_unique RANGE），
        #       按 id 取节点是索引命中，成本恒定；
        #   (2) 别名不会再参与 CONTAINS/属性分支，避免了"名字包含别名"把菜谱节点拉进来
        #       （实测名为「无骨鸡爪」「香辣鸡爪煲」的菜谱度数 9~27，会挤掉度数为 3 的食材种子）；
        #   (3) 实测成本：别名走全部分支 1076~1527ms，按 nodeId 定位约等于原词单次解析的开销。
        alias_ids: List[str] = []
        for alias in alias_names:
            alias_ids.extend(str(nid) for nid in self._name_index.get(alias, []))
        alias_ids = _dedup_preserve_order(alias_ids)[:_ALIAS_MAX_NODE_IDS]

        cypher = """
        // 两条分支合并成一次解析：
        //   分支 A = 用户原词，保持原有三级命中语义（名字精确 3 / 名字包含 2 / 属性命中 1）不变；
        //   分支 B = 同族别名，按 nodeId 精确定位到 :Ingredient，统一记为命中层级 2。
        // 用 UNION 而不是把别名并入同一个 UNWIND：别名是召回辅助而非用户原话，
        // 让它只按"节点名完全等于别名"这一种方式命中，既不放大扫描成本，也不引入子串噪声。
        CALL {
            UNWIND $names AS kw
            MATCH (n)
            WHERE (n:Recipe OR n:Ingredient OR n:CookingStep OR n:Category)
              AND n.nodeId IS NOT NULL
              AND NOT coalesce(n.nodeId, '') IN $placeholder_ids
              AND (
                    toLower(coalesce(n.name, '')) = kw
                 OR toLower(coalesce(n.name, '')) CONTAINS kw
                 OR toLower(coalesce(n.cuisineType, '')) CONTAINS kw
                 OR toLower(coalesce(n.category, '')) CONTAINS kw
                 OR toLower(coalesce(n.tags, '')) CONTAINS kw
                 OR toLower(coalesce(n.description, '')) CONTAINS kw
              )
            WITH DISTINCT n,
                 CASE
                     WHEN toLower(coalesce(n.name, '')) = kw THEN 3
                     WHEN toLower(coalesce(n.name, '')) CONTAINS kw THEN 2
                     ELSE 1
                 END AS match_level
            RETURN n, match_level
            UNION
            UNWIND $alias_ids AS aid
            MATCH (n:Ingredient)
            WHERE n.nodeId = aid
              AND NOT coalesce(n.nodeId, '') IN $placeholder_ids
            WITH DISTINCT n, 2 AS match_level
            RETURN n, match_level
        }
        WITH n, match_level,
             // 「汇总型」名字（如「无骨肉（猪肉、牛肉、鸡肉、鱼丸、火腿肠）」「肉类（牛羊鱼虾等）」）
             // 是抽取阶段产生的聚合条目，名字里带括号/顿号/斜杠。它们不是干净的食材实体，
             // 同层级内排到最后，避免靠度数（聚合条目度数往往不低）挤掉真实食材。
             CASE WHEN coalesce(n.name, '') CONTAINS '（'
                    OR coalesce(n.name, '') CONTAINS '、'
                    OR coalesce(n.name, '') CONTAINS '/'
                  THEN 1 ELSE 0 END AS name_is_aggregate,
             COUNT { (n)--() } AS degree
        ORDER BY match_level DESC, name_is_aggregate ASC, degree DESC, coalesce(n.nodeId, '')
        LIMIT $limit
        RETURN n.nodeId AS id, n.name AS name, labels(n) AS labels,
               properties(n) AS properties, degree, match_level, name_is_aggregate
        """

        params = {
            "names": [n.lower() for n in names],
            "alias_ids": alias_ids,
            "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
            "limit": max(1, int(limit)),
        }

        rows: List[Dict[str, Any]] = []
        try:
            if session is not None:
                rows = [dict(record) for record in session.run(cypher, params)]
            else:
                rows = self._run_cypher(cypher, params)
        except Exception as e:
            logger.error(f"种子节点解析失败: {e}")
            rows = []

        # 【为什么 Cypher 里要写 `n.nodeId IS NOT NULL`】
        # 白名单里的 Category 节点【nodeId 全为 NULL】（实测 count(c)=19 / count(c.nodeId)=0），
        # 而本模块下游一律用 nodeId 定位种子（阶段一 `WHERE seed.nodeId IN $seed_ids`、
        # _select_expansion_seeds 的 `if s.get("id")`），无 nodeId 的种子即使保留下来也
        # 无法被遍历 —— 唯一实际后果是它在 LIMIT 里白占一个名额：
        # 实测 `_resolve_seed_nodes(['荤菜'])` 只返回 4 个菜谱种子（LIMIT=5 被一个
        # nodeId 为 NULL 的 Category 行吃掉了 1 个名额）。因此在 Cypher 侧直接过滤，
        # 让 LIMIT 名额全部留给可遍历的真实种子；Python 侧的真值判断同时保留，
        # 作为对"未来某类节点又开始返回 NULL nodeId"的兜底防御。
        seeds: List[Dict[str, Any]] = []
        for row in self._select_seed_rows(rows, limit):
            seed = {
                "id": str(row.get("id") or ""),
                "node_id": str(row.get("id") or ""),
                "name": str(row.get("name") or ""),
                "labels": list(row.get("labels") or []),
                "label": (list(row.get("labels") or []) or [""])[0],
                "properties": dict(row.get("properties") or {}),
                "degree": int(row.get("degree") or 0),
            }
            if seed["id"]:
                seeds.append(seed)

        # 图谱侧没命中时，退回内存索引（预热过的 entity_cache）做一次模糊解析，尽量不让链路空转
        if not seeds:
            seeds = self._resolve_seed_nodes_from_cache(names, limit)
        return seeds

    def _select_seed_rows(self, rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """从候选行里收口出最终种子行（名称证据优先）。

        唯一的规则是【名称证据优先于属性证据】：
        命中层级 3/2 都是"节点名命中"，层级 1 是「菜系/分类/标签/描述」这类属性命中。
        属性命中度数偏高（烹饪步骤节点度数 8~11），而食材节点度数恒为 3，
        因此在「match_level DESC, degree DESC」的排序下，属性命中会在名称命中耗尽后
        立刻占满剩余名额 —— 实测 `_resolve_seed_nodes(['鸡肉'])` 的 5 个种子里
        有 3 个是「意式烤鸡」(菜谱) 与「步骤4/步骤5」(烹饪步骤)，真正的鸡肉食材只占 2 个。
        所以：存在名称级命中时直接丢弃属性级命中；只有当整批候选都没有名称级命中
        （例如『川菜』只存在于 cuisineType、`resolve_seed_nodes(['荤菜'])` 只能靠 category）时，
        才使用属性级命中 —— 这时丢弃它们会让解析彻底落空。

        【为什么不做"名字多样性"收口】曾尝试"每个名字取 1 个代表"来提高种子铺开度，
        实测发现对反规范化图谱是有害的：同名节点的每一个都对应【不同菜谱】，
        它们是彼此独立的遍历起点，不是冗余。以『土豆』(16 个同名节点) 为例，
        取 5 个「土豆」节点 = 覆盖 5 个土豆菜谱；而"多样性"收口会换成
        「土豆 + 土豆淀粉 + 土豆干粉条 + 小土豆 + 土豆炖排骨」，实际只覆盖 1 个土豆菜谱。
        别名扩展（_expand_alias_names）才是提升覆盖面的正确手段 ——
        它增加的是【名字】这一维，而不是在同一名字内部做取舍。

        Args:
            rows: Cypher 返回的候选行（已按 match_level DESC, degree DESC, nodeId 升序）
            limit: 最终种子数上限

        Returns:
            收口后的候选行（长度不超过 limit，保序、确定性）
        """
        if not rows:
            return []

        limit = max(1, int(limit))
        name_rows = [r for r in rows if int(r.get("match_level") or 0) >= 2]
        if name_rows and len(name_rows) < len(rows):
            logger.info(
                f"种子候选收口：名称级命中 {len(name_rows)} 行，"
                f"丢弃 {len(rows) - len(name_rows)} 行属性级命中（名称证据优先）"
            )
        return (name_rows or list(rows))[:limit]

    def _resolve_seed_nodes_from_cache(self, names: List[str], limit: int) -> List[Dict[str, Any]]:
        """内存兜底：当图谱查询失败或未命中时，用预热的 entity_cache 做实体解析。

        这是降级路径的一环——即使 Neo4j 查询临时不可用，只要索引已经预热，检索仍能给出可用结果。
        """
        scored: List[Tuple[int, int, str, Dict[str, Any]]] = []
        for node_id, info in self.entity_cache.items():
            node_name = str(info.get("name") or "")
            if not node_name or node_name in _PLACEHOLDER_NAMES:
                continue
            lowered = node_name.lower()
            for name in names:
                if lowered == name:
                    scored.append((3, int(info.get("degree") or 0), node_id, info))
                    break
                if name and name in lowered:
                    scored.append((2, int(info.get("degree") or 0), node_id, info))
                    break
        scored.sort(key=lambda item: (-item[0], -item[1]))
        seeds: List[Dict[str, Any]] = []
        for _, degree, node_id, info in scored[:limit]:
            seeds.append({
                "id": str(node_id),
                "node_id": str(node_id),
                "name": str(info.get("name") or ""),
                "labels": list(info.get("labels") or []),
                "label": (list(info.get("labels") or []) or [""])[0],
                "properties": {"name": info.get("name"), "category": info.get("category")},
                "degree": degree,
            })
        if seeds:
            logger.info(f"种子节点解析走内存兜底，命中 {len(seeds)} 个")
        return seeds
