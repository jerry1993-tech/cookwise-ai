"""
查询路由回归测试：不依赖 Neo4j / Milvus / LLM。

覆盖：
1. 18 条真实业务用例的规则兜底路由
2. tips 知识库（相克 / 食品安全 / 配菜规则）
3. 策略层的不检索硬门控
4. 多轮约束延续
5. 证据覆盖校验
6. Web 层对闲聊 / 澄清 / 有证据三种 response_type 的处理
7. 追问分类与澄清配置
8. 口味/否定偏好槽位与候选过滤 Cypher
9. jieba 关键词分词与标点分块兜底
10. LLM 契约只保留语义字段，路由元数据由 RouteKeySpec 派生
11. graph_entity 路由分发覆盖与回答规则配置
"""

import json
import logging
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 规则兜底会逐条打 warning；测试时只保留错误日志
logging.getLogger("rag_modules.query_routing.understanding").setLevel(logging.ERROR)

from langchain_core.documents import Document

import rag_modules.query_routing.text_utils as text_utils
from rag_modules.query_routing import (
    DOMAIN,
    EvidenceValidator,
    GraphEvidenceAdapter,
    QueryUnderstandingService,
    RouteDecisionLogger,
    RoutePolicy,
    RouteThreshold,
    TipsKnowledgeBase,
)
from rag_modules.query_routing.schemas import ContextInfo, EntityRef, IntentItem, SubPlan, Understanding
from rag_modules.query_routing.text_utils import fallback_keywords, segment_words
from rag_modules.intelligent_query_router import IntelligentQueryRouter
from rag_modules.session_cache_manager import SessionCacheManager
from rag_modules.web_service_handler import WebServiceHandler


# ---------------------------------------------------------------------------
# 1. 18 条用例
# ---------------------------------------------------------------------------

CASES = [
    ("清蒸鲈鱼要大火蒸几分钟？具体怎么做？", "recipe_steps_time", True),
    ("做宫保鸡丁需要准备哪些食材和调料？", "recipe_ingredients", True),
    ("可乐鸡翅的做法是什么？", "recipe_overview", True),  # 用户口径：做法是什么 -> 概览
    ("菠菜和豆腐能一起吃吗？为什么？", "ingredient_pairing", True),
    ("胡萝卜和白萝卜一起吃会损失什么？", "ingredient_pairing", True),
    ("牛奶和巧克力能不能一起吃？", "ingredient_pairing", True),
    ("减脂期有什么低热量的菜可以推荐？", "recommend_list", True),
    ("我是厨房新手，想做几道简单快手菜，推荐一下？", "recommend_list", True),
    ("8个人在家聚餐，荤素怎么配？要加鱼吗？", "recommend_scenario", True),
    ("麻辣减脂荞麦面一份大概有多少千卡？", "nutrition_value", True),
    ("羊肉汤有什么营养和功效，适合什么情况喝？", "nutrition_benefit", True),
    ("儿童常见的食物过敏原有哪些？", "food_safety", True),
    ("你觉得宫保鸡丁和红烧肉哪个更好吃？", "chitchat", False),
    ("今天突然好想吃点甜的呀～", "chitchat", False),
    ("你自己会在家做饭吗？", "chitchat", False),
    ("北京明天天气怎么样？", "out_of_scope", False),
    ("推荐几部最近好看的电影吧", "out_of_scope", False),
    ("微信账号怎么注册？", "out_of_scope", False),
]


def test_rule_understanding():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    policy = RoutePolicy(DOMAIN)

    for query, expected_route, expected_retrieval in CASES:
        understanding = service.understand(query)
        plan = policy.plan(understanding, top_k=5)
        routes = [item.intent.route_key for item in plan.sub_plans]
        assert expected_route in routes, f"{query} -> {routes}, 期望 {expected_route}"
        assert plan.needs_retrieval is expected_retrieval, (
            f"{query} needs_retrieval={plan.needs_retrieval}, 期望 {expected_retrieval}"
        )

    print("PASS: 18 条用例规则路由")


def test_slot_vocabulary():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")

    assert service._extract_slots("减脂期的快手菜")["diet"] == "减脂"
    assert service._extract_slots("给儿童吃的")["persona"] == "儿童"
    assert service._extract_slots("不吃辣")["spicy"] == "不辣"
    assert service._extract_slots("推荐几道素菜")["category"] == "素菜"
    assert service._extract_slots("8个人聚餐要加鱼吗")["add_fish"] is True

    assert service._pure_constraint("我不吃辣").route_key == "chitchat"
    assert service._pure_constraint("我不吃辣，推荐几道菜") is None

    print("PASS: 槽位词表")


def test_taste_and_exclude_slots():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")

    # LLM 的 flavor/taste/spicy 三种写法都归一到 spicy
    assert service._normalize_slots({"flavor": "麻辣"}, "recommend_list")["spicy"] == "辣"
    assert service._normalize_slots({"taste": "中辣口味"}, "recommend_list")["spicy"] == "中辣"
    assert service._normalize_slots({"taste": "甜味"}, "recommend_list")["taste"] == "甜"
    # 问菜谱详情时，口味只是菜名描述，不作为过滤槽位
    assert "spicy" not in service._normalize_slots({"flavor": "麻辣"}, "nutrition_value")

    assert service._extract_slots("推荐几道辣口的菜")["spicy"] == "辣"
    assert "spicy" not in service._extract_slots("推荐麻辣香锅")
    assert service._extract_slots("那哪些菜是带辣椒的")["ingredient"] == "辣椒"
    assert service._extract_slots("带甜味的菜")["flavor"] == "甜"
    assert service._extract_slots("不喜欢吃甜的") == {"exclude": ["甜"]}
    assert service._extract_slots("不要放甜品") == {"exclude": ["甜品"]}
    assert service._extract_slots("不放辣椒的菜") == {"exclude": ["辣椒"]}
    assert service._extract_slots("不要辣的菜")["spicy"] == "不辣"

    # 纯偏好声明不检索，但要把排除项写进会话状态
    item = service._pure_constraint("我不喜欢香菜")
    assert item is not None and item.slots["exclude"] == ["香菜"]

    # 第二轮否定偏好要累加，不能覆盖上一轮的排除项
    context = {"constraints": {"exclude": ["香菜"]}}
    item = service._pure_constraint("我不喜欢甜品")
    understanding = service._constraint_understanding("我不喜欢甜品", context, item)
    assert understanding.intents[0].slots["exclude"] == ["香菜", "甜品"]

    # 规则兜底路径也要把上一轮否定偏好带进推荐意图
    understanding = service.understand("推荐几道菜", context=context)
    assert understanding.intents[0].slots["exclude"] == ["香菜"]

    print("PASS: 口味与否定偏好槽位")


def test_candidate_filters():
    adapter = GraphEvidenceAdapter()
    captured = {}

    def fake_run(cypher, params=None):
        captured["cypher"] = cypher
        captured["params"] = params or {}
        return []

    adapter._run = fake_run

    adapter._run_candidate_query({"spicy": "辣"}, 8)
    assert ":Ingredient" in captured["cypher"] and "REQUIRES" in captured["cypher"]
    assert "NOT (" not in captured["cypher"]
    assert "辣" in captured["params"].values()

    adapter._run_candidate_query({"spicy": "不辣"}, 8)
    assert "NOT (" in captured["cypher"]
    assert "辣" in captured["params"].values()

    adapter._run_candidate_query({"ingredient": "辣椒"}, 8)
    assert "REQUIRES" in captured["cypher"]
    assert "辣椒" in captured["params"].values()

    # 食材同义词（西红柿/番茄）会一起展开
    adapter._run_candidate_query({"ingredient": "西红柿"}, 8)
    values = {value for value in captured["params"].values() if isinstance(value, str)}
    assert {"西红柿", "番茄"} <= values

    adapter._run_candidate_query({"exclude": ["香菜", "甜品"]}, 8)
    assert captured["cypher"].count("NOT (") == 2
    values = {value for value in captured["params"].values() if isinstance(value, str)}
    assert {"香菜", "甜品"} <= values

    print("PASS: 候选过滤 Cypher")


def test_jieba_keyword_fallback():
    words = segment_words("推荐几道低热量适合新手的家常菜")
    assert "低热量" in words
    assert "家常菜" in words
    assert "推荐几道低热量适合新手的家常菜" not in words  # 不再把整句当一个词

    entities, topics = fallback_keywords(
        "水煮鱼低热量适合新手",
        recipes=["水煮鱼"],
        topic_terms=("低热量", "新手"),
    )
    assert entities == ["水煮鱼"]
    assert "低热量" in topics

    # jieba 不可用时仍退回标点分块，保证兜底不崩
    original = text_utils.jieba
    text_utils.jieba = None
    try:
        assert text_utils.segment_words("水煮鱼，低热量") == ["水煮鱼", "低热量"]
    finally:
        text_utils.jieba = original

    print("PASS: jieba 关键词分词")


def test_llm_contract_uses_route_spec_metadata():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")

    for route_key, span in (
        ("recommend_list", "推荐几道菜"),
        ("recipe_ingredients", "水煮鱼要哪些食材"),
    ):
        spec = DOMAIN.spec(route_key)
        parsed = {
            "intents": [{
                "intent": "out_of_scope",  # 故意写错，验证以 route_key 为准
                "route_key": route_key,
                "confidence": 0.9,
                "span": span,
                "entities": [],
                "slots": {},
                "evidence_profile": ["NONE"],
                "need_complete": not spec.need_complete,
                "value_policy": "oops",
                "expected_source": ["fake"],
            }]
        }
        understanding = service._from_dict(parsed, span, None, source="llm")
        item = understanding.intents[0]
        assert item.intent == spec.intent
        assert item.evidence_profile == list(spec.evidence_profile)
        assert item.need_complete == spec.need_complete
        assert item.value_policy == spec.value_policy
        assert item.expected_source == [spec.source]

    print("PASS: LLM 契约只保留语义字段")


def test_graph_entity_dispatch_covers_all_route_specs():
    from rag_modules.query_routing.domain import ROUTE_SPECS
    from rag_modules.query_routing.evidence import EvidenceService

    class FakeGraph:
        def __getattr__(self, name):
            def handler(*args, **kwargs):
                return [Document(page_content=name, metadata={"evidence_type": name})]
            return handler

    class FakeText:
        def search(self, *args, **kwargs):
            return []

        def recipe_search(self, *args, **kwargs):
            return []

    service = EvidenceService(FakeGraph(), FakeText(), tips=None)
    for key, spec in ROUTE_SPECS.items():
        if spec.engine != "graph_entity":
            continue
        item = IntentItem(intent=spec.intent, route_key=key)
        plan = SubPlan(
            intent=item,
            query="测试",
            engine="graph_entity",
            top_k=5,
            need_complete=spec.need_complete,
            validator=spec.validator,
        )
        assert service.execute(plan), f"{key} 没有 graph_entity 处理分支"

    print("PASS: graph_entity 路由分发覆盖")


def test_answer_rules_use_known_routes():
    from rag_modules.query_routing.domain import ROUTE_SPECS
    from rag_modules.query_routing.prompts import ANSWER_RULES

    assert ANSWER_RULES
    assert set(ANSWER_RULES) <= set(ROUTE_SPECS)
    for key in ANSWER_RULES:
        assert ROUTE_SPECS[key].engine != "none"

    print("PASS: 回答规则配置")


def test_follow_up_rules():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    context = {"active_recipe": "宫保鸡丁"}

    item = service._classify_follow_up("它需要哪些食材？", context)
    assert item.route_key == "recipe_ingredients"

    item = service._classify_follow_up("那要多少盐？", context)
    assert item.route_key == "recipe_ingredients"
    assert item.slots["ingredient"] == "盐"

    item = service._classify_follow_up("要煮多久？", context)
    assert item.route_key == "recipe_steps_time"

    item = service._classify_follow_up("它有多少千卡？", context)
    assert item.route_key == "nutrition_value"

    item = service._classify_follow_up("换成牛肉可以吗？", context)
    assert item.route_key == "ingredient_property"

    # 没有指代/省略词时不走追问分类，交给普通规则
    assert service._classify_follow_up("需要哪些食材？", context) is None

    # 兜底：指代 + “做法”归概览
    item = service._classify_follow_up("这道菜的做法是什么？", context)
    assert item.route_key == "recipe_overview"

    print("PASS: 追问分类配置")


def test_clarify_config():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    default_spec = DOMAIN.spec(DOMAIN.default_route_key)

    reference = service._reference_clarify("它好吃吗？")
    assert reference.intents[0].route_key == default_spec.key
    assert reference.intents[0].intent == default_spec.intent
    assert reference.clarify_question == DOMAIN.clarify_messages["reference_missing"]

    empty = service.understand("")
    assert empty.intents[0].route_key == DOMAIN.empty_query_key
    assert empty.clarify_question == DOMAIN.clarify_messages["empty_query"]

    legacy = service._legacy_understanding({"recommended_strategy": "graph_rag"}, "宫保鸡丁", None)
    assert legacy.intents[0].intent == default_spec.intent

    print("PASS: 澄清配置")


# ---------------------------------------------------------------------------
# 2. tips 知识库
# ---------------------------------------------------------------------------


def test_complete_partial_recipe_name():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    service.set_known_entities(
        recipes=["空气炸锅面包片", "电饭煲蒸米饭", "可乐鸡翅"],
        ingredients=["面包片", "米饭", "鸡翅", "牛奶"],
    )
    # LLM 只截取了菜名的一部分：用原句里的完整菜名补全
    entities = service._normalize_entities(
        [{"name": "面包片", "type": "ingredient"}],
        "空气炸锅面包片的热量高吗？",
    )
    assert [(item.name, item.type) for item in entities] == [("空气炸锅面包片", "recipe")]

    # 原句里没有完整菜名时保持原样，避免把普通食材误升级成菜谱
    entities = service._normalize_entities(
        [{"name": "牛奶", "type": "ingredient"}], "牛奶怎么加热？"
    )
    assert [(item.name, item.type) for item in entities] == [("牛奶", "ingredient")]


def test_mislabeled_entity_type_is_corrected():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    service.set_known_entities(
        recipes=["无骨鸡爪", "可乐鸡翅"],
        ingredients=["鸡翅", "牛奶"],
    )
    # 菜名被 LLM 标成食材：词表里只在菜谱侧存在，纠正为菜谱
    entities = service._normalize_entities(
        [{"name": "无骨鸡爪", "type": "ingredient"}], "无骨鸡爪有什么营养？"
    )
    assert [(item.name, item.type) for item in entities] == [("无骨鸡爪", "recipe")]

    # 食材被标成菜谱：同样按词表纠正
    entities = service._normalize_entities(
        [{"name": "牛奶", "type": "recipe"}], "牛奶怎么加热？"
    )
    assert [(item.name, item.type) for item in entities] == [("牛奶", "ingredient")]


def test_invalid_route_key_uses_intent():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    parsed = {
        "standalone_query": "讲一个睡前故事",
        "needs_retrieval": False,
        "intents": [{
            "intent": "out_of_scope",
            "route_key": "none",
            "confidence": 0.98,
            "span": "讲一个睡前故事",
        }],
    }
    understanding = service._from_dict(parsed, "讲一个睡前故事", None, source="test")
    assert understanding is not None
    assert [item.route_key for item in understanding.intents] == ["out_of_scope"]
    assert understanding.needs_retrieval is False


def test_history_intents_are_not_repeated():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    query = "家里有2个人，推荐一桌菜"
    history = "用户: 巴沙鱼怎么去腥？另外，有没有闽菜推荐？"
    parsed = {
        "standalone_query": "家里有2个人，推荐一桌菜；巴沙鱼怎么去腥；有没有闽菜推荐",
        "needs_retrieval": True,
        "intents": [
            {"intent": "recommendation", "route_key": "recommend_scenario",
             "confidence": 0.92, "span": query},
            {"intent": "ingredient_info", "route_key": "ingredient_property",
             "confidence": 0.95, "span": "巴沙鱼怎么去腥"},
            {"intent": "recommendation", "route_key": "recommend_list",
             "confidence": 0.9, "span": "有没有闽菜推荐"},
        ],
    }
    understanding = service._from_dict(
        parsed, query, None, source="test", history_text=history
    )
    assert understanding is not None
    assert [item.route_key for item in understanding.intents] == ["recommend_scenario"]

    # 本轮带指代词时，即使 span 来自历史也要保留（多轮改写场景）
    parsed["intents"] = [{
        "intent": "recipe_detail", "route_key": "recipe_steps_time",
        "confidence": 0.9, "span": "宫保鸡丁要煮多久",
    }]
    understanding = service._from_dict(
        parsed, "它要煮多久？", None, source="test",
        history_text="用户: 宫保鸡丁要煮多久？",
    )
    assert understanding is not None
    assert [item.route_key for item in understanding.intents] == ["recipe_steps_time"]


def test_value_measure_matches_requested_unit():
    markdown = (
        "# 电饭煲蒸米饭\n\n"
        "* 江南米，水量约 2.6 厘米。\n"
        "* 烹饪时间 30-40分钟。\n"
        "* 热量 254 千卡。\n"
    )
    # 问热量时只取热量，不能拿菜名里的“蒸”触发的时间数字顶替
    value, unit_type, _ = GraphEvidenceAdapter.find_measure(
        markdown, "电饭煲蒸米饭有多少千卡？"
    )
    assert (value, unit_type) == ("254 千卡", "kcal")
    assert GraphEvidenceAdapter.find_measure(
        "烹饪时间 30-40分钟", "电饭煲蒸米饭有多少千卡？"
    ) is None
    # 问时间时仍按时间取
    time_value, time_type, _ = GraphEvidenceAdapter.find_measure(markdown, "要煮多久？")
    assert time_type == "time" and time_value.endswith("分钟")


def test_tips_knowledge():
    kb = TipsKnowledgeBase()

    pairing = kb.lookup_pairing("菠菜和豆腐能一起吃吗？")
    assert pairing is not None
    assert pairing.left == "菠菜" and pairing.right == "豆腐"
    assert "草酸钙" in pairing.principle

    safety = kb.lookup_safety("儿童常见的食物过敏原有哪些？")
    assert safety is not None and safety.subgroup == "儿童"
    assert "花生" in safety.items and "蛋类" in safety.items

    plan8 = kb.meal_plan(8)
    assert (plan8.dishes, plan8.meat, plan8.veg, plan8.add_fish) == (9, 5, 4, False)

    plan9 = kb.meal_plan(9)
    assert plan9.add_fish is True

    technique = kb.lookup_technique("鱼怎么去腥？")
    assert technique is not None and "去腥" in technique.title

    print("PASS: tips 知识库")


def test_same_route_multi_recipe():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    service.set_known_entities(recipes=["宫保鸡丁", "可乐鸡翅"])
    policy = RoutePolicy(DOMAIN)

    understanding = service.understand("宫保鸡丁要哪些食材？可乐鸡翅要哪些食材？")
    plan = policy.plan(understanding, top_k=5)

    assert len(plan.sub_plans) == 2
    assert plan.sub_plans[0].intent.entity_names("recipe")[0] == "宫保鸡丁"
    assert plan.sub_plans[1].intent.entity_names("recipe")[0] == "可乐鸡翅"

    print("PASS: 同路由键多菜名不合并")


# ---------------------------------------------------------------------------
# 3. 多轮约束延续
# ---------------------------------------------------------------------------


def test_multi_turn_constraint():
    service = QueryUnderstandingService(llm_client=None, llm_model="fake")
    context = {"constraints": {"spicy": "不辣"}, "recent_user_queries": ["我不吃辣"]}

    understanding = service.understand("水煮鱼怎么做", context=context)
    assert understanding.intents[0].slots.get("spicy") == "不辣"

    # 纯约束声明不检索，只记录约束
    understanding = service.understand("我不吃辣")
    assert understanding.intents[0].route_key == "chitchat"
    assert understanding.needs_retrieval is False
    assert understanding.intents[0].slots.get("spicy") == "不辣"

    # 没有 active_recipe 的指代追问先澄清，避免全局检索乱答
    understanding = service.understand("它需要哪些食材？")
    assert understanding.is_ambiguous is True
    assert understanding.needs_retrieval is False

    print("PASS: 多轮约束延续")


# ---------------------------------------------------------------------------
# 4. 会话状态
# ---------------------------------------------------------------------------


def test_session_state():
    manager = SessionCacheManager(embedding_model=None)
    analysis = SimpleNamespace(
        understanding={"intents": [{"entities": [{"name": "宫保鸡丁", "type": "recipe"}]}]},
        topic_shift=False,
        route_key="recipe_ingredients",
        topic_id="宫保鸡丁",
        slots={"spicy": "不辣"},
        intent="recipe_detail",
    )
    manager.update_state_after_turn("s1", "宫保鸡丁要哪些食材", analysis, "ok")

    state = manager.get_state("s1")
    assert state["active_recipe"] == "宫保鸡丁"
    assert state["constraints"]["spicy"] == "不辣"
    assert manager.get_context_for_query("s1", "那要多少盐") == "宫保鸡丁 那要多少盐"

    shift = SimpleNamespace(
        understanding={"intents": []},
        topic_shift=True,
        route_key="chitchat",
        topic_id="chitchat",
        slots={},
        intent="chitchat",
    )
    manager.update_state_after_turn("s1", "换个话题", shift, "ok")
    assert manager.get_state("s1")["active_recipe"] == ""

    # 域外问题视为话题中断：旧菜名清空，后续指代不能复用
    manager.update_state_after_turn("s1", "宫保鸡丁要哪些食材", analysis, "ok")
    out_of_scope = SimpleNamespace(
        understanding={"intents": []},
        topic_shift=False,
        route_key="out_of_scope",
        topic_id="out_of_scope",
        slots={},
        intent="out_of_scope",
    )
    manager.update_state_after_turn("s1", "北京明天天气怎么样？", out_of_scope, "ok")
    assert manager.get_state("s1")["active_recipe"] == ""
    assert manager.get_state("s1")["active_topic"] == "out_of_scope"

    print("PASS: 会话状态")


# ---------------------------------------------------------------------------
# 5. 主题键
# ---------------------------------------------------------------------------


def test_topic_id():
    context = ContextInfo(active_recipe="宫保鸡丁")
    base = Understanding(standalone_query="", needs_retrieval=True, intents=[], context=context)

    recommend = IntentItem(intent="recommendation", route_key="recommend_list")
    assert IntelligentQueryRouter._topic_id(base, recommend, DOMAIN) == "recommend_list"

    steps = IntentItem(intent="recipe_detail", route_key="recipe_steps_time")
    assert IntelligentQueryRouter._topic_id(base, steps, DOMAIN) == "宫保鸡丁"

    pairing = IntentItem(
        intent="ingredient_info",
        route_key="ingredient_pairing",
        entities=[EntityRef("菠菜", "ingredient"), EntityRef("豆腐", "ingredient")],
    )
    assert IntelligentQueryRouter._topic_id(base, pairing, DOMAIN) == "菠菜+豆腐"

    print("PASS: 主题键")


def test_top_intent_fallback():
    empty = Understanding(standalone_query="", needs_retrieval=False, intents=[])
    item = empty.top_intent("recipe_detail", "recipe_overview")
    assert item.intent == "recipe_detail"
    assert item.route_key == "recipe_overview"

    print("PASS: 空意图默认路由")


# ---------------------------------------------------------------------------
# 6. 证据覆盖校验
# ---------------------------------------------------------------------------


def test_validator():
    validator = EvidenceValidator()
    intent = IntentItem(intent="recipe_detail", route_key="recipe_ingredients")
    plan = SubPlan(intent=intent, query="水煮鱼要哪些食材", engine="graph_entity", top_k=20,
                   need_complete=True, validator="list")

    complete_doc = Document(
        page_content="食材清单",
        metadata={"evidence_type": "ingredients", "expected_count": 10, "found_count": 10},
    )
    result = validator.validate(plan, [complete_doc])
    assert result.status == "complete"

    partial_doc = Document(
        page_content="食材清单",
        metadata={"evidence_type": "ingredients", "expected_count": 10, "found_count": 8},
    )
    result = validator.validate(plan, [partial_doc])
    assert result.status == "partial"

    value_intent = IntentItem(intent="nutrition_health", route_key="nutrition_value")
    value_plan = SubPlan(intent=value_intent, query="多少千卡", engine="graph_entity", top_k=1,
                         need_complete=False, validator="value")
    assert validator.validate(value_plan, []).status == "missing"

    print("PASS: 证据覆盖校验")


def test_validator_registry_covers_all_route_specs():
    from rag_modules.query_routing.domain import ROUTE_SPECS
    from rag_modules.query_routing.validators import VALIDATORS

    for key, spec in ROUTE_SPECS.items():
        if spec.engine == "none":
            assert spec.validator == "", key
        else:
            assert spec.validator in VALIDATORS, key


# ---------------------------------------------------------------------------
# 7. 按路由键阈值
# ---------------------------------------------------------------------------


def test_route_thresholds():
    understanding = Understanding(
        standalone_query="水煮鱼要哪些食材",
        needs_retrieval=True,
        intents=[IntentItem(intent="recipe_detail", route_key="recipe_ingredients", confidence=0.6)],
    )
    assert RoutePolicy(DOMAIN).plan(understanding).needs_retrieval is True

    strict = replace(
        DOMAIN,
        route_thresholds={"default": RouteThreshold(min_execute=0.8)},
    )
    assert RoutePolicy(strict).plan(understanding).response_type == "clarify"

    # 检索覆盖阈值：LLM 说“不检索”，但检索意图置信度够高时仍执行
    no_retrieval_understanding = Understanding(
        standalone_query="水煮鱼要哪些食材",
        needs_retrieval=False,
        intents=[IntentItem(intent="recipe_detail", route_key="recipe_ingredients", confidence=0.75)],
    )
    assert RoutePolicy(DOMAIN).plan(no_retrieval_understanding).needs_retrieval is True
    strict_override = replace(
        DOMAIN,
        route_thresholds={"default": RouteThreshold(retrieval_override=0.9)},
    )
    assert RoutePolicy(strict_override).plan(no_retrieval_understanding).needs_retrieval is False

    # 闲聊拦截阈值：混合意图时，闲聊置信度够高可以压掉检索
    mixed = Understanding(
        standalone_query="谢谢，水煮鱼要哪些食材",
        needs_retrieval=True,
        intents=[
            IntentItem(intent="chitchat", route_key="chitchat", confidence=0.85),
            IntentItem(intent="recipe_detail", route_key="recipe_ingredients", confidence=0.6),
        ],
    )
    assert RoutePolicy(DOMAIN).plan(mixed).needs_retrieval is False
    strict_chitchat = replace(
        DOMAIN,
        route_thresholds={"default": RouteThreshold(no_retrieval=0.9)},
    )
    assert RoutePolicy(strict_chitchat).plan(mixed).needs_retrieval is True

    print("PASS: 按路由键阈值")


# ---------------------------------------------------------------------------
# 8. 路由决策落盘
# ---------------------------------------------------------------------------


def test_route_decision_logger():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "route_decisions.jsonl"
        logger = RouteDecisionLogger(path=str(path), enabled=True)
        logger.log({"route_key": "recipe_ingredients", "confidence": 0.8})
        logger.log({"route_key": "chitchat", "confidence": 0.9})

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["route_key"] == "recipe_ingredients"
        assert first["confidence"] == 0.8
        assert first["ts"]

    print("PASS: 路由决策落盘")


# ---------------------------------------------------------------------------
# 5. Web 层三种 response_type
# ---------------------------------------------------------------------------


class FakeCacheManager:
    def check_semantic_cache(self, query, session_id):
        return None

    def get_state(self, session_id):
        return {"constraints": {}, "active_topic": "", "active_recipe": "",
                "recent_user_queries": []}

    def add_to_semantic_cache(self, query, response, session_id):
        pass

    def add_to_context(self, session_id, query, response):
        pass

    def update_state_after_turn(self, session_id, query, analysis, response=""):
        pass

    def get_session_stats(self):
        return {}


class FakeRouter:
    mode = "chitchat"

    def route_query(self, query, top_k=5, session_id="", session_state=None):
        if self.mode == "clarify":
            analysis = SimpleNamespace(
                response_type="clarify",
                clarify_question="你想问哪道菜？",
                needs_retrieval=False,
                standalone_query=query,
                route_key="recipe_overview",
                intent="recipe_detail",
                topic_id="",
                topic_shift=False,
                slots={},
                understanding={},
            )
            return [], analysis

        if self.mode == "answer":
            analysis = SimpleNamespace(
                response_type="answer",
                clarify_question="",
                needs_retrieval=True,
                standalone_query=query,
                route_key="recipe_ingredients",
                intent="recipe_detail",
                topic_id="水煮鱼",
                topic_shift=False,
                slots={},
                understanding={},
            )
            return [Document(page_content="水煮鱼食材证据")], analysis

        analysis = SimpleNamespace(
            response_type="answer",
            clarify_question="",
            needs_retrieval=False,
            standalone_query=query,
            route_key="chitchat",
            intent="chitchat",
            topic_id="",
            topic_shift=False,
            slots={},
            understanding={},
        )
        return [], analysis

    def get_route_statistics(self):
        return {}


class FakeGeneration:
    def generate_adaptive_answer(self, question, documents, analysis=None):
        return "证据回答"

    def generate_adaptive_answer_stream(self, question, documents, analysis=None, max_retries=3):
        yield "证据回答"

    def generate_direct_answer(self, question, analysis=None):
        return "闲聊回答"

    def generate_direct_answer_stream(self, question, analysis=None):
        yield "闲聊回答"


class FakeRecipeManager:
    def get_random_recipes_with_images(self, limit=3):
        return []

    def get_recipe_by_id(self, recipe_id):
        return {}


class FakeConfig:
    top_k = 5


class FakeRagSystem:
    def __init__(self, mode):
        self.cache_manager = FakeCacheManager()
        self.query_router = FakeRouter()
        self.query_router.mode = mode
        self.generation_module = FakeGeneration()
        self.recipe_manager = FakeRecipeManager()
        self.config = FakeConfig()


def test_web_response_types():
    for mode, expected in (("chitchat", "闲聊回答"), ("clarify", "你想问哪道菜？"), ("answer", "证据回答")):
        handler = WebServiceHandler(FakeRagSystem(mode))
        app = handler.setup_flask_app()
        client = app.test_client()
        response = client.post("/api/chat", json={"message": "测试", "session_id": "s1"})
        assert response.status_code == 200
        assert response.get_json()["response"] == expected, f"{mode} -> {response.get_json()}"

    print("PASS: Web 层 response_type 处理")


def test_feedback_endpoint(tmp_path, monkeypatch):
    """反馈接口：点赞落盘、非法请求拒绝、统计可读。"""
    feedback_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(feedback_path))

    handler = WebServiceHandler(FakeRagSystem("answer"))
    client = handler.setup_flask_app().test_client()

    response = client.post("/api/feedback", json={
        "session_id": "s1",
        "query": "宫保鸡丁怎么做？",
        "answer": "先腌鸡丁。",
        "rating": "like",
    })
    assert response.status_code == 200

    invalid = client.post("/api/feedback", json={"query": "没有评分"})
    assert invalid.status_code == 400

    stats = client.get("/api/feedback/stats").get_json()
    assert stats == {"like": 1, "dislike": 0, "correction": 0, "total": 1}
    assert feedback_path.exists()

    print("PASS: 反馈接口")


def test_generation_thinking_disabled():
    """生成调用必须关闭 deepseek-flash 的思考，避免 reasoning 吃光 token。"""
    from rag_modules.generation_integration import GenerationIntegrationModule

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    module = object.__new__(GenerationIntegrationModule)
    completions = FakeCompletions()
    module.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    assert module._create_completion(model="m", messages=[]).choices[0].message.content == "ok"
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}

    print("PASS: 生成关闭思考")


def test_multi_intent_generation_prompt():
    """多意图 Prompt 必须列出每个子问题，并给证据标上所属子问题。"""
    from rag_modules.generation_integration import GenerationIntegrationModule

    module = object.__new__(GenerationIntegrationModule)
    analysis = SimpleNamespace(
        route_key="recipe_ingredients",
        is_multi_intent=True,
        sub_queries=[
            {"route_key": "recipe_ingredients", "query": "宫保鸡丁需要哪些食材？"},
            {"route_key": "recipe_steps_time", "query": "要煮多久？"},
        ],
        slots={},
        disclaimer_required=False,
    )
    prompt = module._build_prompt("宫保鸡丁需要哪些食材？要煮多久？", "证据", analysis)
    assert "必须逐个回答" in prompt
    assert "2. 要煮多久？" in prompt
    # 兜底路径同样要求编号连续，不能每答一个子问题就重新从 1 开始
    assert "不要重新从 1 开始" in prompt
    # 每个子问题应用自己的回答规则，而不是把主路由键的规则套到整篇
    assert "各子问题的回答要求" in prompt
    assert "子问题 2（要煮多久？）：按步骤顺序回答" in prompt
    assert prompt.count("按步骤顺序回答") == 1

    # 只有“怎么做”类问句才追加分条格式；问时间不加，避免答成整套步骤
    steps_analysis = SimpleNamespace(
        route_key="recipe_steps_time",
        is_multi_intent=False,
        sub_queries=[],
        slots={},
        disclaimer_required=False,
    )
    assert "每一步单独一行" in module._build_prompt("水煮鱼怎么做？", "证据", steps_analysis)
    assert "每一步单独一行" not in module._build_prompt("水煮鱼要煮多久？", "证据", steps_analysis)

    docs = [Document(page_content="步骤证据", metadata={"source": "graph", "sub_query": "要煮多久？"})]
    context = module._format_context(docs, analysis)
    assert "子问题：要煮多久？" in context

    print("PASS: 多意图生成 Prompt")


def test_multi_intent_answer_numbering_is_code_owned():
    """多意图回答的节标题与编号由代码生成，模型复述标题也不会出现两个 1.。"""
    from rag_modules.generation_integration import GenerationIntegrationModule

    class FakeCompletions:
        def __init__(self):
            self.prompts = []
            self.responses = [
                "1. 宫保鸡丁需要哪些食材？\n- 鸡肉\n- 花生",
                "1. 要煮多久？\n- 10 分钟",
            ]

        def create(self, **kwargs):
            self.prompts.append(kwargs["messages"][0]["content"])
            content = self.responses[len(self.prompts) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    module = object.__new__(GenerationIntegrationModule)
    completions = FakeCompletions()
    module.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    module.model_name = "fake"
    module.temperature = 0.1
    module.max_tokens = 512

    analysis = SimpleNamespace(
        route_key="recipe_ingredients",
        is_multi_intent=True,
        sub_queries=[
            {"index": 0, "route_key": "recipe_ingredients", "query": "宫保鸡丁需要哪些食材？"},
            {"index": 1, "route_key": "recipe_steps_time", "query": "要煮多久？"},
        ],
        slots={},
        disclaimer_required=False,
    )
    documents = [
        Document(page_content="食材证据", metadata={"sub_query_index": 0}),
        Document(page_content="时间证据", metadata={"sub_query_index": 1}),
    ]

    answer = "".join(
        module.generate_adaptive_answer_stream("宫保鸡丁需要哪些食材？要煮多久？", documents, analysis=analysis)
    )

    assert len(completions.prompts) == 2
    assert "食材证据" in completions.prompts[0] and "时间证据" not in completions.prompts[0]
    assert "时间证据" in completions.prompts[1] and "食材证据" not in completions.prompts[1]
    # 节标题必须是 markdown 标题块；裸 `1. …` 会被渲染成列表项并与正文有序列表合并
    assert answer.startswith("### 1. 宫保鸡丁需要哪些食材？\n")
    assert "\n### 2. 要煮多久？\n" in answer
    assert "\n1. 宫保鸡丁需要哪些食材？" not in answer
    assert "\n2. 要煮多久？" not in answer
    # 模型复述的“1. 要煮多久？”被去掉，编号只由代码写一次
    assert answer.count("1. 宫保鸡丁需要哪些食材？") == 1
    assert answer.count("要煮多久？") == 1
    assert "1. 要煮多久？" not in answer

    completions.responses = ["食材", "时间"]
    completions.prompts.clear()
    text = module.generate_adaptive_answer("宫保鸡丁需要哪些食材？要煮多久？", documents, analysis=analysis)
    assert text.startswith("### 1. 宫保鸡丁需要哪些食材？\n\n食材")
    assert "\n\n### 2. 要煮多久？\n\n时间" in text

    print("PASS: 多意图回答编号由代码生成")


def test_steps_answer_structure_is_repaired():
    """步骤回答被模型压成一段时，按图谱证据重新分条，且不误伤单点时间问题。"""
    from rag_modules.generation_integration import GenerationIntegrationModule

    steps_doc = Document(
        page_content="\n".join([
            "【菜谱】水煮鱼",
            "【图谱步骤（共 3 步）】",
            "1. 步骤1",
            "   描述：解冻鱼片。",
            "   时间：5小时",
            "2. 步骤2",
            "   描述：切片。",
            "3. 步骤3",
            "   描述：盛盘。",
        ]),
        metadata={"evidence_type": "steps"},
    )
    prose = "解冻鱼片。切片。盛盘。"
    repair = GenerationIntegrationModule._ensure_step_structure

    assert repair("水煮鱼怎么做？", prose, [steps_doc]) == (
        "1. 解冻鱼片。（5小时）\n2. 切片。\n3. 盛盘。"
    )
    # 非“怎么做”类问题不展开，避免把“要煮多久”答成整套步骤
    assert repair("要煮多久？", prose, [steps_doc]) == prose
    # 已经分条的回答原样返回
    structured = "1. 解冻\n2. 切片\n3. 盛盘"
    assert repair("水煮鱼怎么做？", structured, [steps_doc]) == structured
    # 证据不足 3 步时不重建
    short_doc = Document(
        page_content="1. 步骤1\n   描述：解冻。\n2. 步骤2\n   描述：切片。",
        metadata={"evidence_type": "steps"},
    )
    assert repair("水煮鱼怎么做？", prose, [short_doc]) == prose
    # 整篇多意图回答不重建，避免把食材等其它子问题的内容替换掉
    multi = SimpleNamespace(is_multi_intent=True)
    assert repair("水煮鱼需要什么食材？怎么做？", prose, [steps_doc], multi) == prose

    print("PASS: 步骤回答结构兜底")


def test_web_stream():
    handler = WebServiceHandler(FakeRagSystem("chitchat"))
    app = handler.setup_flask_app()
    client = app.test_client()
    response = client.post("/api/chat_stream", json={"message": "测试", "session_id": "s2"})
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "闲聊回答" in body
    assert "[DONE]" in body

    print("PASS: Web 流式回答")


if __name__ == "__main__":
    test_rule_understanding()
    test_slot_vocabulary()
    test_taste_and_exclude_slots()
    test_candidate_filters()
    test_jieba_keyword_fallback()
    test_llm_contract_uses_route_spec_metadata()
    test_graph_entity_dispatch_covers_all_route_specs()
    test_answer_rules_use_known_routes()
    test_follow_up_rules()
    test_clarify_config()
    test_tips_knowledge()
    test_same_route_multi_recipe()
    test_multi_turn_constraint()
    test_session_state()
    test_topic_id()
    test_top_intent_fallback()
    test_validator()
    test_route_thresholds()
    test_route_decision_logger()
    test_web_response_types()
    test_generation_thinking_disabled()
    test_multi_intent_generation_prompt()
    test_web_stream()
    print("\n全部查询路由测试通过")
