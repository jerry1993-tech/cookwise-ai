"""查询理解：自然语言 → GraphQuery（LLM + 本地降级）、复杂度分析与自适应查询规划。"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .shared_constants_helpers_models import (
    GraphQuery,
    QueryType,
    _COMPLEXITY_MEDIUM,
    _COMPLEXITY_SIMPLE,
    _COMPLEXITY_WEIGHT_CONNECTIVE,
    _COMPLEXITY_WEIGHT_LENGTH,
    _COMPLEXITY_WEIGHT_MULTIHOP,
    _COMPLEXITY_WEIGHT_PUNCT,
    _CONNECTIVE_CUES,
    _MAX_SUBGRAPH_NODES,
    _MAX_TRAVERSAL_DEPTH,
    _MIN_SUBGRAPH_NODES,
    _MULTI_HOP_PATTERNS,
    _QUERY_CACHE_LIMIT,
    _clamp,
    _coerce_query_text,
    _dedup_preserve_order,
)

logger = logging.getLogger(__name__)


class QueryUnderstandingMixin:
    """查询理解与规划：把自然语言转成受约束的 GraphQuery（LLM + 本地降级），评估复杂度并生成检索计划。"""

    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        """

        # 【入口防御】本方法是公开入口（测试脚本、上层编排都会直接调它，不经过 graph_rag_search），
        # 因此必须自己收口参数类型：此前 int/float/list/dict/bool 都会在内部的 re 处理上抛
        # TypeError: expected string or bytes-like object（实测六种非字符串输入里五种复现）。
        # 检索语义是"降级不抛错"，这里统一转成字符串再走原链路；
        # 放在最前面而不是只保护缓存键，是为了让下面 prompt 拼接、降级分支、
        # _build_graph_query_from_llm 全部拿到同一份规范化文本。
        query = _coerce_query_text(query)

        prompt = f"""作为图数据库专家，分析以下查询的图结构意图：

        查询：{query}

        请识别：
        1. 查询类型：
           - entity_relation: 询问实体间的直接关系（如：鸡肉和胡萝卜能一起做菜吗？）
           - multi_hop: 需要多跳推理（如：鸡肉配什么蔬菜？需要：鸡肉→菜品→食材→蔬菜）
           - subgraph: 需要完整子图（如：川菜有什么特色？需要川菜相关的完整知识网络）
           - path_finding: 路径查找（如：从食材到成品菜的制作路径）
           - clustering: 聚类相似性（如：和宫保鸡丁类似的菜有哪些？）

        2. 核心实体：查询中的关键实体名称
        3. 目标实体：期望找到的实体类型
        4. 关系类型：涉及的关系类型
        5. 遍历深度：需要的图遍历深度（1-3跳）

        -----
        示例：
        查询："鸡肉配什么蔬菜好？"
        分析：这是multi_hop查询，需要通过"鸡肉→使用鸡肉的菜品→这些菜品使用的蔬菜"的路径推理

        返回JSON格式：
        {{
            "query_type": "multi_hop",
            "source_entities": ["鸡肉"],
            "target_entities": ["蔬菜类食材"],
            "relation_types": ["REQUIRES", "BELONGS_TO_CATEGORY"],
            "max_depth": 3,
            "reasoning": "需要多跳推理：鸡肉→菜品→食材→蔬菜"
        }}
        -----

        请严格按照JSON格式返回，不要添加额外解释。
        """

        # 缓存命中：同一问题常被反复问到（路由器一次、生成一次），避免重复的 LLM 调用与 Token 消耗
        # 用 _coerce_query_text 而不是 `(query or "")`：后者对 int/list/dict 会抛 AttributeError
        cache_key = _coerce_query_text(query).strip()
        cached = self._query_understanding_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"意图理解命中缓存: {cache_key[:40]}")
            return cached

        result = self._call_llm_json(prompt)

        if result:
            graph_query = self._build_graph_query_from_llm(result, query)
        else:
            # 降级方案：LLM 不可用/返回不可解析时，用本地启发式推断查询类型并解析真实实体名
            logger.warning("意图理解走本地降级路径")
            graph_query = self._fallback_graph_query(query)

        # 写入缓存（超过容量则清空重来，避免引入 LRU 依赖；该缓存只用于省调用，丢失无副作用）
        if len(self._query_understanding_cache) >= _QUERY_CACHE_LIMIT:
            self._query_understanding_cache.clear()
        self._query_understanding_cache[cache_key] = graph_query
        return graph_query

    def _call_llm_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """调用 LLM 并解析出 JSON 对象；任何失败都返回 None（由调用方决定降级策略）。

        三个必须处理的现实问题：
          1. 【推理模型】deepseek-flash 会先吐 reasoning_content，必须 extra_body 关闭思考，否则 content 是空串；
          2. 【代码围栏】模型常把 JSON 包在 ```json ... ``` 里，需要剥离；
          3. 【前后缀噪声】偶尔会带一句解释，需要从文本中提取最外层 JSON 对象。
        """
        if self.llm_client is None:
            logger.warning("LLM 客户端为空，跳过 LLM 调用")
            return None

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}}   # 关键：关闭思考
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None

        if not content:
            # 关闭思考后仍拿到空内容，说明模型行为异常；记日志并交给降级路径
            logger.warning("LLM 返回空内容，转入降级路径")
            return None

        parsed = self._extract_json_object(content)
        if parsed is None:
            logger.warning(f"LLM 输出无法解析为 JSON: {content[:160]}")
        return parsed

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """从模型输出中稳健地提取 JSON 对象。

        依次尝试：直接解析 -> 剥掉代码围栏后解析 -> 截取最外层花括号区间解析。
        """
        if not text:
            return None

        candidates: List[str] = [text.strip()]

        # 剥离 ```json ... ``` / ``` ... ``` 代码围栏
        fenced = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        # 截取第一个 { 到最后一个 } 之间的内容（容忍模型在 JSON 前后加解释）
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            candidates.append(text[first:last + 1].strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _coerce_str_list(value: Any) -> List[str]:
        """把 LLM 返回的任意形态字段规范化为「字符串列表」，且绝不抛异常。

        为什么不能只写 `result.get(k) or []`：LLM 返回的 JSON 形状完全不受控。
        实测（用一个会返回指定 payload 的假客户端喂进 understand_graph_query）：
          - {"source_entities": 5}         -> 直接 for 迭代 int 抛 TypeError，
                                              而 understand_graph_query 是在
                                              graph_rag_search 的 try 之外被调用的，
                                              异常会一路逃出公开入口；
          - {"source_entities": {"a": 1}}  -> dict 被迭代成键 ['a']，静默变成错误实体；
          - {"relation_types": "REQUIRES"} -> 字符串被逐字符拆成 ['R','E','Q',...]，
                                              关系类型过滤静默失效（不报错，只是白写）。

        因此这里按类型显式分流，宁可丢弃也不猜：
          - None / 数值 / 布尔 -> []（无效，丢弃；由调用方走本地兜底）
          - str               -> [该字符串]（单个实体名是合法的；逐字符拆才是错的）
          - dict              -> []（键不是实体名，宁缺毋滥）
          - list / tuple / set-> 逐元素转字符串并去空白，非标量元素丢弃
        集合类型的迭代顺序不确定，但调用方随后会做去重，顺序无影响。
        """
        if value is None or isinstance(value, bool):
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, dict):
            return []
        if isinstance(value, (int, float)):
            return []
        if isinstance(value, (list, tuple, set)):
            out: List[str] = []
            for item in value:
                if isinstance(item, (str, int, float)) and not isinstance(item, bool):
                    text = str(item).strip()
                    if text:
                        out.append(text)
            return out
        return []

    def _build_graph_query_from_llm(self, result: Dict[str, Any], query: str) -> GraphQuery:
        """把 LLM 返回的 JSON 规范化为受约束的 GraphQuery。

        四道防御：
          - query_type 非法值兜底为 SUBGRAPH（而不是让 QueryType(...) 直接抛异常打断检索）；
          - source_entities / target_entities / relation_types 一律经 _coerce_str_list 规范化，
            杜绝 LLM 返回标量或 dict 时抛 TypeError / 静默产出错误实体（详见该方法的说明）；
          - max_depth 走 _clamp_int 钳制到 [1, _MAX_TRAVERSAL_DEPTH]，杜绝 LLM 返回 10 导致 Cypher 爆炸；
          - max_nodes 钳制到 [_MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES]；
          - source_entities 为空时用本地实体抽取补上，避免拿到空起点。
        """
        try:
            query_type = QueryType(str(result.get("query_type", "")).strip().lower())
        except (ValueError, AttributeError):
            logger.warning(f"LLM 返回了非法的 query_type: {result.get('query_type')!r}，兜底为 subgraph")
            query_type = QueryType.SUBGRAPH

        source_entities = [
            self._normalize_entity_name(e) for e in self._coerce_str_list(result.get("source_entities"))
        ]
        source_entities = [e for e in _dedup_preserve_order(source_entities) if e]
        if not source_entities:
            source_entities = self._candidate_entities_from_query(query)

        target_entities = [
            self._normalize_entity_name(e) for e in self._coerce_str_list(result.get("target_entities"))
        ]
        target_entities = [e for e in _dedup_preserve_order(target_entities) if e]

        # 关系类型只保留图谱里真实存在的，避免 ANY(... IN $relation_types) 因拼写错误恒为 false
        relation_types = [
            str(r).strip().upper() for r in self._coerce_str_list(result.get("relation_types"))
            if str(r).strip()
        ]
        relation_types = [r for r in _dedup_preserve_order(relation_types) if r]

        return GraphQuery(
            query_type=query_type,
            source_entities=source_entities,
            target_entities=target_entities,
            relation_types=relation_types,
            max_depth=_clamp(result.get("max_depth", 2), 1, _MAX_TRAVERSAL_DEPTH, 2),
            max_nodes=_clamp(result.get("max_nodes", 50), _MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES, 50),
        )

    def _fallback_graph_query(self, query: str) -> GraphQuery:
        """LLM 不可用时的本地降级：用关键词模式推断查询类型，并抽取真实实体名。

        相比「统一兜底成 SUBGRAPH + source_entities=[query]」，这里至少能：
          - 用模式匹配区分出 CLUSTERING / MULTI_HOP / PATH_FINDING，走到更合适的分支；
          - 把实体解析成图谱里真实存在的名字，让后续遍历有起点。
        """
        text = query or ""
        if re.search(r"(类似|相似|同款|一样的)", text):
            query_type = QueryType.CLUSTERING
        elif re.search(r"(路径|怎么从|如何从|经过哪些|流程|先后)", text):
            query_type = QueryType.PATH_FINDING
        elif any(re.search(p, text) for p in _MULTI_HOP_PATTERNS):
            query_type = QueryType.MULTI_HOP
        elif re.search(r"(关系|为什么|区别)", text):
            query_type = QueryType.ENTITY_RELATION
        else:
            query_type = QueryType.SUBGRAPH

        depth = 3 if query_type in (QueryType.MULTI_HOP, QueryType.PATH_FINDING) else 2
        return GraphQuery(
            query_type=query_type,
            source_entities=self._candidate_entities_from_query(query),
            max_depth=depth,
            max_nodes=50,
        )

    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应查询规划：根据查询复杂度动态调整策略
        """
        # 分析查询复杂度
        complexity_score = self._analyze_query_complexity(query)

        # 先把查询解析成图谱里真实存在的实体名，避免把整句查询当成实体导致全程空转
        candidates = self._candidate_entities_from_query(query)

        query_plans = []

        if complexity_score < _COMPLEXITY_SIMPLE:
            # 简单查询：直接邻居查询
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=candidates,
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)

        elif complexity_score < _COMPLEXITY_MEDIUM:
            # 中等复杂度：多跳查询
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=candidates,
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)

        else:
            # 复杂查询：子图提取 + 推理
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=candidates,
                max_depth=3,
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=candidates,
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])

        return query_plans

    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度，返回 [0,1] 的分数，供 adaptive_query_planning 选择遍历深度。

        四个加权分量（合计 1.0），设计目标是让「简单实体查询」与「多跳推理查询」拉开可观测的差距：
          - 长度分量 (0.25)：长查询通常承载更多约束；
          - 多跳模式分量 (0.35)：命中 _MULTI_HOP_PATTERNS（如「配什么」「为什么」「类似」）越多越复杂；
          - 多实体连接词分量 (0.30)：出现「和 / 跟 / 还是」等说明在同时谈多个对象；
          - 标点分量 (0.10)：多分句、多问句代表复合意图。
        该分数还与 GraphRAGConfig.max_graph_depth 的语义保持一致（上限同为 3 跳）。
        """
        text = _coerce_query_text(query)
        if not text.strip():
            return 0.0

        # 1) 长度分量：以 60 字为饱和点
        length_score = min(len(text) / 60.0, 1.0) * _COMPLEXITY_WEIGHT_LENGTH

        # 2) 多跳模式分量：命中 2 个即视为饱和
        multihop_hits = sum(1 for pattern in _MULTI_HOP_PATTERNS if re.search(pattern, text))
        multihop_score = min(multihop_hits / 2.0, 1.0) * _COMPLEXITY_WEIGHT_MULTIHOP

        # 3) 多实体连接词分量：命中 2 个即视为饱和
        connective_hits = sum(1 for cue in _CONNECTIVE_CUES if cue in text)
        connective_score = min(connective_hits / 2.0, 1.0) * _COMPLEXITY_WEIGHT_CONNECTIVE

        # 4) 标点分量：逗号/顿号/问号累计 3 个即视为饱和
        punct_hits = text.count("，") + text.count("、") + text.count("？") + text.count("?") + text.count("；")
        punct_score = min(punct_hits / 3.0, 1.0) * _COMPLEXITY_WEIGHT_PUNCT

        return round(min(length_score + multihop_score + connective_score + punct_score, 1.0), 4)
