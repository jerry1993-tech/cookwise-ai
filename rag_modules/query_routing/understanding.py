"""
理解层：一次 LLM 调用完成“指代消解 + 多意图识别 + 槽位抽取”。

关键取舍：
- LLM 只产出结构化理解，不产出检索策略；
- 解析失败先修复一次，再失败走本地规则，不让整条链路崩掉；
- 规则兜底用图谱实体词典，不用 ``query.split()``；
- 业务词表全部来自 domain.py；槽位归一化/抽取在 slots.py，
  属于领域适配层，换领域时需同步替换。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from .domain import DOMAIN, DomainConfig
from .prompts import build_repair_prompt, build_understanding_prompt
from .rules import RuleClassifier
from .schemas import ContextInfo, EntityRef, IntentItem, Understanding
from .slots import SlotExtractor
from .text_utils import (
    clamp_float,
    dedupe,
    extract_known_entities,
    has_reference,
    normalize_text,
    split_segments,
)

logger = logging.getLogger(__name__)

# 没有上下文时需要澄清的指代词；“几分钟”“多少盐”这类单独出现不算
ANAPHORA_TERMS = ("它", "这个", "那个", "这道", "那这道", "这菜", "那菜")


class QueryUnderstandingService:
    """把自然语言问题变成 :class:`Understanding`。"""

    def __init__(
        self,
        llm_client,
        llm_model: str,
        domain: DomainConfig = DOMAIN,
        known_entities: Optional[Dict[str, Sequence[str]]] = None,
        cache_size: int = 128,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.domain = domain
        self.cache_size = cache_size
        self._cache: Dict[str, Understanding] = {}

        known_entities = known_entities or {}
        self.recipes = list(known_entities.get("recipes", []))
        self.ingredients = list(known_entities.get("ingredients", []))
        self.cuisines = list(known_entities.get("cuisines", []))
        self._recipe_set = set(self.recipes)
        self._ingredient_set = set(self.ingredients)
        self._cuisine_set = set(self.cuisines)
        self.slot_extractor = SlotExtractor(domain, self.cuisines)
        self.rule_classifier = RuleClassifier(
            domain, self._extract_entity_refs, self.slot_extractor
        )

    def set_known_entities(
        self,
        recipes: Sequence[str] = (),
        ingredients: Sequence[str] = (),
        cuisines: Sequence[str] = (),
    ) -> None:
        """数据加载完成后刷新实体词表；路由器的首次查询会调用一次。"""
        self.recipes = list(recipes)
        self.ingredients = list(ingredients)
        self.cuisines = list(cuisines)
        self._recipe_set = set(self.recipes)
        self._ingredient_set = set(self.ingredients)
        self._cuisine_set = set(self.cuisines)
        self.slot_extractor.set_cuisines(self.cuisines)
        self._cache.clear()  # 词表变了，旧理解结果可能已经过期

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def understand(
        self,
        query: str,
        history_text: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Understanding:
        """理解一条查询；任何异常都收敛到规则兜底。"""
        query = normalize_text(query)
        if not query:
            return self._empty_understanding()

        constraints_key = json.dumps(
            (context or {}).get("constraints", {}), sort_keys=True, ensure_ascii=False
        )
        cache_key = (
            f"{query}\n--\n{history_text}\n--\n"
            f"{(context or {}).get('active_recipe', '')}\n--\n{constraints_key}"
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # 纯约束声明（“我不吃辣”）只更新会话状态，不让 LLM 误判成推荐
        constraint_item = self._pure_constraint(query)
        if constraint_item is not None:
            return self._remember(
                cache_key, self._constraint_understanding(query, context, constraint_item)
            )

        # 有指代（“它…/这道…”）但会话里没有可绑定的菜名：先澄清，避免全局检索乱答
        has_recipe_in_query = any(
            entity.type == "recipe" for entity in self._extract_entity_refs(query)
        )
        if (
            not (context or {}).get("active_recipe")
            and not has_recipe_in_query
            and has_reference(query, ANAPHORA_TERMS)
        ):
            return self._remember(cache_key, self._reference_clarify(query))

        prompt = build_understanding_prompt(query, history_text, self.domain)
        parsed = self._call_llm_json(prompt)

        if parsed:
            understanding = self._from_dict(
                parsed, query, context, source="llm", history_text=history_text
            )
            if understanding is not None:
                return self._remember(cache_key, understanding)

            # 第一次 JSON 能解析但字段不合格 -> 带错误信息修复一次
            repair_prompt = build_repair_prompt(
                raw_output=json.dumps(parsed, ensure_ascii=False),
                error="intents 为空、route_key 非法或字段类型不符合要求",
                domain=self.domain,
            )
            repaired = self._call_llm_json(repair_prompt)
            if repaired:
                understanding = self._from_dict(
                    repaired, query, context, source="llm_repair", history_text=history_text
                )
                if understanding is not None:
                    return self._remember(cache_key, understanding)

        logger.warning("理解层走规则兜底: %s", query)
        understanding = self.rule_understand(query, context, history_text)
        return self._remember(cache_key, understanding)

    def rule_understand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history_text: str = "",
    ) -> Understanding:
        """不依赖 LLM 的规则理解，也是所有失败路径的最终兜底。"""
        query = normalize_text(query)
        context = context or {}

        segments = split_segments(query)
        items: List[IntentItem] = []
        for segment in segments:
            # 有 active_recipe 的追问（“那要多少盐”）优先按上下文补全
            item = (
                self.rule_classifier.classify_follow_up(segment, context)
                or self.rule_classifier.rule_classify(segment)
            )
            if item is None:
                # “为什么？”“还有呢？”这类短追问并入上一段，避免丢上下文
                if items and len(segment) <= 6:
                    items[-1].span = f"{items[-1].span}；{segment}"
                continue
            items.append(item)

        items = self.rule_classifier.merge_intents(items)
        if not items:
            items = [self.rule_classifier.default_intent(query)]

        # 上一轮声明的约束（如“不吃辣”）延续到本轮
        for item in items:
            for key, value in (context.get("constraints") or {}).items():
                if key == "exclude":
                    item.slots["exclude"] = self._merge_exclude(
                        item.slots.get("exclude"), value
                    )
                else:
                    item.slots.setdefault(key, value)
        self._attach_active_recipe(items, context)

        standalone_query = self._standalone_query(query, context)
        context_info = self._context_info(query, context, items)
        needs_retrieval = any(
            item.route_key not in self.domain.no_retrieval_keys for item in items
        )

        return Understanding(
            standalone_query=standalone_query,
            needs_retrieval=needs_retrieval,
            intents=items,
            context=context_info,
            disclaimer_required=any(
                self.domain.spec(item.route_key).disclaimer for item in items
            ),
            source="rule",
        )

    # ------------------------------------------------------------------
    # LLM 调用与解析
    # ------------------------------------------------------------------

    def _call_llm_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """调用 LLM 并解析 JSON；失败返回 None。"""
        if self.llm_client is None:
            return None

        kwargs = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1200,
            "stream": False,
        }

        try:
            try:
                # DeepSeek 推理模型需要显式关闭 thinking，否则 content 可能为空
                response = self.llm_client.chat.completions.create(
                    **kwargs,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except TypeError:
                # 兼容不支持 extra_body 的 OpenAI 兼容客户端
                response = self.llm_client.chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("理解层 LLM 调用失败: %s", exc)
            return None

        if not content:
            return None
        return self._extract_json_object(content)

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """从模型输出中提取 JSON：直接解析 -> 去代码围栏 -> 截最外层花括号。"""
        if not text:
            return None

        candidates = [text.strip()]
        fenced = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

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

    def _from_dict(
        self,
        parsed: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]],
        source: str,
        history_text: str = "",
    ) -> Optional[Understanding]:
        """把 LLM JSON 规范化为 Understanding；不合格返回 None。"""
        raw_intents = parsed.get("intents")
        if not isinstance(raw_intents, list) or not raw_intents:
            # 兼容旧版路由 Prompt 的返回格式（只有 recommended_strategy）
            return self._legacy_understanding(parsed, query, context)

        items: List[IntentItem] = []
        for raw in raw_intents:
            if not isinstance(raw, dict):
                continue
            item = self._parse_intent(raw, query, context, history_text, parsed)
            if item is not None:
                items.append(item)

        if not items:
            return None
        self._attach_active_recipe(items, context or {})
        return self._build_understanding(parsed, query, context, source, items)

    def _parse_intent(
        self,
        raw: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]],
        history_text: str,
        parsed: Dict[str, Any],
    ) -> Optional[IntentItem]:
        """把单个 LLM 意图规范化为 IntentItem；无效返回 None。"""
        span = str(raw.get("span") or query).strip()
        # LLM 偶尔把历史轮次的意图重复算进本轮；本轮没有指代词时丢弃纯历史 span
        if self._from_history_only(span, query, history_text):
            return None

        route_key = str(raw.get("route_key") or "").strip()
        if route_key not in self.domain.route_specs:
            # LLM 可能返回 "none"；闲聊/域外这种单路由键意图直接按意图映射
            intent_keys = self.domain.intent_to_keys.get(
                str(raw.get("intent") or "").strip(), ()
            )
            if len(intent_keys) == 1:
                route_key = intent_keys[0]
            else:
                inferred = self.rule_classifier.infer_route_key(span)
                if inferred not in self.domain.route_specs:
                    return None
                route_key = inferred
        # “X 的做法是什么”这类措辞按领域配置改写路由
        route_key = self.rule_classifier.apply_wording_rules(route_key, span)

        spec = self.domain.spec(route_key)

        entities = self._normalize_entities(raw.get("entities"), span)
        slots = raw.get("slots") if isinstance(raw.get("slots"), dict) else {}
        if not slots and isinstance(parsed.get("constraints"), dict):
            slots = parsed["constraints"]
        # LLM 可能用 people_count / goal 等字段名；统一成系统内部的槽位键
        slots = self._normalize_slots(slots, route_key)
        for key, value in self._extract_slots(span).items():
            slots.setdefault(key, value)
        for key, value in ((context or {}).get("constraints") or {}).items():
            if key == "exclude":
                # 否定偏好跨轮累加，而不是被本轮的空列表覆盖
                slots["exclude"] = self._merge_exclude(slots.get("exclude"), value)
            else:
                slots.setdefault(key, value)

        # route_key 确定后，证据形态/完整性/来源都以 RouteKeySpec 为准；
        # LLM 只负责语义理解，避免它和配置给出两套不一致的策略。
        return IntentItem(
            intent=spec.intent,
            route_key=route_key,
            evidence_profile=list(spec.evidence_profile),
            confidence=clamp_float(raw.get("confidence"), 0.0, 1.0, 0.6),
            span=span,
            entities=entities,
            slots=slots,
            need_complete=spec.need_complete,
            value_policy=spec.value_policy,
            expected_source=[spec.source],
            engine_override=str(raw.get("engine_override") or ""),
        )

    def _build_understanding(
        self,
        parsed: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]],
        source: str,
        items: List[IntentItem],
    ) -> Understanding:
        """汇总意图、上下文与安全标记，构造 Understanding。"""
        needs_retrieval = parsed.get("needs_retrieval")
        if not isinstance(needs_retrieval, bool):
            needs_retrieval = any(
                item.route_key not in self.domain.no_retrieval_keys for item in items
            )

        standalone_query = str(parsed.get("standalone_query") or "").strip()
        if not standalone_query:
            standalone_query = self._standalone_query(query, context or {})

        context_info = self._context_info(query, context or {}, items, parsed.get("context"))
        ambiguity = parsed.get("ambiguity") if isinstance(parsed.get("ambiguity"), dict) else {}
        safety = parsed.get("safety") if isinstance(parsed.get("safety"), dict) else {}

        return Understanding(
            standalone_query=standalone_query,
            needs_retrieval=needs_retrieval,
            intents=items,
            context=context_info,
            is_ambiguous=bool(ambiguity.get("is_ambiguous", False)),
            clarify_question=str(ambiguity.get("clarify_question") or ""),
            disclaimer_required=bool(
                safety.get("disclaimer_required", False)
                or any(self.domain.spec(item.route_key).disclaimer for item in items)
            ),
            source=source,
            raw=parsed,
        )

    def _from_history_only(self, span: str, query: str, history_text: str) -> bool:
        """意图只存在于历史轮次（span 在历史里、不在本轮问题里），且本轮没有指代词。"""
        if not history_text or not span or span in query or span not in history_text:
            return False
        has_anaphora = has_reference(query, ANAPHORA_TERMS)
        has_continuation = has_reference(query, self.domain.reference_terms)
        return not (has_anaphora or has_continuation)

    def _legacy_understanding(
        self,
        parsed: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[Understanding]:
        """兼容旧版 Prompt：只有 recommended_strategy 时映射成单个通用意图。"""
        strategy = str(parsed.get("recommended_strategy") or "").strip().lower()
        if strategy not in ("hybrid_traditional", "graph_rag", "combined"):
            return None

        engine_override = {
            "hybrid_traditional": "hybrid",
            "graph_rag": "graph_relation",
            "combined": "combined",
        }[strategy]

        entity_refs = self._extract_entity_refs(query)
        default_spec = self.domain.spec(self.domain.default_route_key)
        item = IntentItem(
            intent=default_spec.intent,
            route_key=default_spec.key,
            evidence_profile=["TEXT"],
            confidence=clamp_float(parsed.get("confidence"), 0.0, 1.0, 0.6),
            span=query,
            entities=entity_refs,
            need_complete=False,
            expected_source=["legacy"],
            engine_override=engine_override,
        )
        return Understanding(
            standalone_query=query,
            needs_retrieval=True,
            intents=[item],
            context=self._context_info(query, context or {}, [item]),
            source="legacy",
            raw=parsed,
        )

    def _remember(self, key: str, understanding: Understanding) -> Understanding:
        """写缓存；超过容量直接清空，避免引入额外 LRU 依赖。"""
        if len(self._cache) >= self.cache_size:
            self._cache.clear()
        self._cache[key] = understanding
        return understanding

    # ------------------------------------------------------------------
    # 规则分类
    # ------------------------------------------------------------------

    # 规则分类放在 rules.py；这两个方法名给旧调用/测试保留。
    def _classify_follow_up(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> Optional[IntentItem]:
        """委托给 RuleClassifier。"""
        return self.rule_classifier.classify_follow_up(text, context)

    def _pure_constraint(self, query: str) -> Optional[IntentItem]:
        """委托给 RuleClassifier。"""
        return self.rule_classifier.pure_constraint(query)

    # ------------------------------------------------------------------
    # 实体 / 槽位 / 上下文
    # ------------------------------------------------------------------

    def _extract_entity_refs(self, text: str) -> List[EntityRef]:
        """用图谱词表抽取实体；词表为空时返回空列表。"""
        matched = extract_known_entities(
            text,
            recipes=self.recipes,
            ingredients=self.ingredients,
            cuisines=self.cuisines,
            aliases=self.domain.aliases,
            generic_names=self.domain.generic_entity_names,
        )
        return [
            EntityRef(
                name=name,
                type=entity_type,
                unresolved=(
                    (entity_type == "recipe" and self._recipe_set and name not in self._recipe_set)
                    or (entity_type == "ingredient" and self._ingredient_set and name not in self._ingredient_set)
                ),
            )
            for name, entity_type in matched
        ]

    def _normalize_entities(self, value: Any, text: str) -> List[EntityRef]:
        """规范化 LLM 返回的 entities；格式不对时退回词典抽取。"""
        if not isinstance(value, list) or not value:
            return self._extract_entity_refs(text)

        entities: List[EntityRef] = []
        seen = set()
        for raw in value:
            if isinstance(raw, dict):
                name = normalize_text(raw.get("name") or raw.get("text") or "")
                entity_type = str(raw.get("type") or "unknown")
                resolved_id = str(raw.get("resolved_id") or "")
            elif isinstance(raw, str):
                name = normalize_text(raw)
                entity_type = "unknown"
                resolved_id = ""
            else:
                continue

            if not name or name in seen or name in self.domain.generic_entity_names:
                continue
            seen.add(name)

            # LLM 可能把菜谱写成 dish / 菜品，统一成内部类型
            type_aliases = {
                "dish": "recipe", "菜品": "recipe", "菜谱": "recipe",
                "recipe_name": "recipe", "food": "ingredient", "食材": "ingredient",
                "ingredient_name": "ingredient", "菜系": "cuisine", "cuisine_type": "cuisine",
            }
            entity_type = type_aliases.get(entity_type, entity_type)

            if entity_type == "unknown":
                entity_type = self._resolve_entity_type(name)
            # LLM 偶发把类型标反；词表里只在一边能查到时纠正
            if (
                entity_type == "ingredient"
                and name in self._recipe_set
                and name not in self._ingredient_set
            ):
                entity_type = "recipe"
            elif (
                entity_type == "recipe"
                and name in self._ingredient_set
                and name not in self._recipe_set
            ):
                entity_type = "ingredient"
            unresolved = entity_type == "unknown"
            if entity_type == "recipe" and self._recipe_set:
                unresolved = name not in self._recipe_set
            elif entity_type == "ingredient" and self._ingredient_set:
                unresolved = name not in self._ingredient_set

            entities.append(EntityRef(
                name=name,
                type=entity_type,
                resolved_id=resolved_id,
                unresolved=unresolved,
            ))

        if entities:
            entities = self._complete_recipe_names(entities, text)
        return entities or self._extract_entity_refs(text)

    def _complete_recipe_names(
        self, entities: List[EntityRef], text: str
    ) -> List[EntityRef]:
        """LLM 偶尔只截取菜名的一部分（“空气炸锅面包片”->“面包片”），用原句里的最长已知菜名补全。"""
        span_recipes = [
            item.name for item in self._extract_entity_refs(text) if item.type == "recipe"
        ]
        completed: List[EntityRef] = []
        seen = set()
        for entity in entities:
            candidates = [
                name for name in span_recipes
                if entity.name != name and entity.name in name
            ]
            longest = max(candidates, key=len) if candidates else ""
            # 有多个并列最长候选时不猜，保留 LLM 原结果
            if longest and sum(1 for name in candidates if len(name) == len(longest)) == 1:
                entity = EntityRef(name=longest, type="recipe", unresolved=False)
            if entity.name not in seen:
                seen.add(entity.name)
                completed.append(entity)
        return completed

    def _resolve_entity_type(self, name: str) -> str:
        """用已知词表判断实体类型。"""
        if name in self._recipe_set:
            return "recipe"
        if name in self._ingredient_set:
            return "ingredient"
        if name in self._cuisine_set:
            return "cuisine"
        return "unknown"

    # 槽位归一化/抽取的具体实现放在 slots.py；这里保留方法名给旧调用和测试。
    def _normalize_slots(self, slots: Any, route_key: str = "") -> Dict[str, Any]:
        """委托给 SlotExtractor。"""
        return self.slot_extractor.normalize(slots, route_key)

    def _extract_slots(self, text: str) -> Dict[str, Any]:
        """委托给 SlotExtractor。"""
        return self.slot_extractor.extract(text)

    @staticmethod
    def _merge_exclude(*groups: Any) -> List[str]:
        """委托给 SlotExtractor 的排除项合并。"""
        return SlotExtractor.merge_exclude(*groups)

    def _attach_active_recipe(self, items: List[IntentItem], context: Dict[str, Any]) -> None:
        """追问没有显式菜名时，把 active_recipe 补进意图实体，供证据层绑定。"""
        active_recipe = str(context.get("active_recipe") or "")
        if not active_recipe:
            return

        for item in items:
            spec = self.domain.spec(item.route_key)
            if spec.entity_first and not item.entity_names("recipe"):
                item.entities.append(EntityRef(name=active_recipe, type="recipe"))

    def _standalone_query(self, query: str, context: Dict[str, Any]) -> str:
        """有指代且状态里有 active_recipe 时，做最小改写；否则原样返回。"""
        active_recipe = str(context.get("active_recipe") or "")
        if active_recipe and has_reference(query, self.domain.reference_terms) and active_recipe not in query:
            return f"{active_recipe} {query}"
        return query

    def _context_info(
        self,
        query: str,
        context: Dict[str, Any],
        items: List[IntentItem],
        llm_context: Optional[Dict[str, Any]] = None,
    ) -> ContextInfo:
        """合并会话状态与 LLM 返回的 context。"""
        llm_context = llm_context if isinstance(llm_context, dict) else {}
        active_recipe = str(llm_context.get("active_recipe") or context.get("active_recipe") or "")
        active_topic = str(llm_context.get("active_topic") or context.get("active_topic") or "")

        topic_shift = bool(llm_context.get("topic_shift", False))
        if any(term in query for term in ("换个话题", "不说这个", "聊点别的")):
            topic_shift = True

        resolved_entities = dedupe(
            [entity.name for item in items for entity in item.entities]
        )

        return ContextInfo(
            used_history=bool(
                llm_context.get("used_history", False)
                or (active_recipe and has_reference(query, self.domain.reference_terms))
            ),
            topic_shift=topic_shift,
            active_recipe=active_recipe,
            active_topic=active_topic,
            resolved_entities=resolved_entities,
        )

    def _constraint_understanding(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        item: IntentItem,
    ) -> Understanding:
        """约束声明的理解结果：不检索，但保留上下文菜名和已有约束。"""
        context = context or {}
        merged_slots = dict(context.get("constraints") or {})
        for key, value in item.slots.items():
            if key == "exclude":
                merged_slots["exclude"] = self._merge_exclude(
                    merged_slots.get("exclude"), value
                )
            else:
                merged_slots[key] = value
        item.slots = merged_slots
        return Understanding(
            standalone_query=query,
            needs_retrieval=False,
            intents=[item],
            context=self._context_info(query, context, [item]),
            source="rule",
        )

    def _reference_clarify(self, query: str) -> Understanding:
        """指代没有可绑定的菜名：返回澄清，不触发检索。"""
        spec = self.domain.spec(self.domain.default_route_key)
        return Understanding(
            standalone_query=query,
            needs_retrieval=False,
            intents=[IntentItem(
                intent=spec.intent,
                route_key=spec.key,
                evidence_profile=["NONE"],
                confidence=0.5,
                span=query,
            )],
            is_ambiguous=True,
            clarify_question=self.domain.clarify_messages["reference_missing"],
            source="rule",
        )

    def _empty_understanding(self) -> Understanding:
        """空查询的安全返回：不检索、直接澄清。"""
        spec = self.domain.spec(self.domain.empty_query_key)
        return Understanding(
            standalone_query="",
            needs_retrieval=False,
            intents=[IntentItem(
                intent=spec.intent,
                route_key=spec.key,
                evidence_profile=["NONE"],
                confidence=0.5,
                span="",
            )],
            is_ambiguous=True,
            clarify_question=self.domain.clarify_messages["empty_query"],
            source="rule",
        )
