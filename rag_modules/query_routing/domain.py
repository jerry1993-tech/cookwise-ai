"""
CookWise 的业务配置：意图体系、路由键、证据源、业务词表、图谱/向量库 Schema。

这是整个路由包里最“业务”的文件。
换到金融 RAG 时，先替换本文件 + prompts.py + evidence.py；
规则兜底、追问分类、槽位抽取、策略层、校验层都由本文件配置驱动。
slots.py 负责槽位归一化/抽取（servings/max_minutes 等），
它属于领域适配层，换领域时需同步替换。
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteKeySpec:
    """一个路由键的静态定义。"""

    key: str
    intent: str
    evidence_profile: Tuple[str, ...]   # LIST / REL / TEXT / VALUE / FILTER / RULE / NONE
    engine: str                         # 执行层使用的引擎名
    source: str                         # 人可读的证据源说明
    need_complete: bool
    default_top_k: int
    validator: str
    keywords: Tuple[str, ...]           # Prompt 示例 + 规则触发词；核心规则词见上方常量

    # 以下是策略层/路由层读取的标志；换领域时只改配置，不写死路由键
    entity_first: bool = False          # 多意图缺菜名时，补共享菜名
    expand_per_recipe: bool = False     # 一个意图含多道菜时，拆成每道菜
    prefix_shared_recipe: bool = True   # 多意图子查询里补共享菜名
    depends_on_previous: Tuple[str, ...] = ()  # 生成顺序依赖的上一路由键
    refuse: bool = False                # 直接拒绝回答（域外）
    reasoning_required: bool = False    # 旧接口的 reasoning_required 标记
    topic_mode: str = "route"           # 主题键策略：route / chitchat / pair
    disclaimer: bool = False            # 回答附免责声明
    value_policy: str = ""              # exact_only 等数值策略
    boost_with_entity: bool = False     # 命中实体时把置信度抬到 0.8


@dataclass(frozen=True)
class RuleSpec:
    """规则兜底的一条规则；按顺序匹配，第一条命中即返回。"""

    route_key: str
    terms: Tuple[str, ...] = ()          # 命中任意一个词就算匹配
    slots: Dict[str, str] = field(default_factory=dict)  # 命中后写入的槽位
    # 从候选词里挑第一个命中的词写入槽位（如追问里的 ingredient）
    slot_from_terms: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    require_request: bool = False        # 还必须出现“要推荐/要建议”的词
    skip_if_request: bool = False        # 出现这类词就不算（“好想吃”不是推荐）
    skip_if_any: Tuple[str, ...] = ()    # 出现任意一个词就跳过本规则


@dataclass(frozen=True)
class WordingRule:
    """措辞改写：命中 terms 且不含 exclude_terms 时，把源路由改成目标路由。"""

    source_keys: Tuple[str, ...]
    target_key: str
    terms: Tuple[str, ...]
    exclude_terms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintSpec:
    """纯约束声明（如“我不吃辣”）的识别配置。"""

    terms: Tuple[str, ...]
    request_terms: Tuple[str, ...]   # 出现这些词就不算纯约束
    route_key: str
    slots: Dict[str, str]
    confidence: float = 0.9


@dataclass(frozen=True)
class CandidateFilterSpec:
    """推荐候选的一个过滤维度；evidence 按字段模板生成 Cypher。"""

    slot: str                                            # 槽位名，如 spicy / ingredient
    fields: Tuple[str, ...] = ("name", "tags")           # 匹配哪些图谱字段
    terms: Tuple[str, ...] = ()                          # 固定过滤词；空则用槽位值
    trigger_values: Tuple[str, ...] = ()                 # 只在槽位取这些值时生效
    negative_values: Tuple[str, ...] = ()                # 这些值表示“排除”
    negative: bool = False                               # 整个槽位都是排除项
    multi_value: bool = False                            # 槽位是列表（如 exclude）
    expand_aliases: bool = False                         # 用 domain.aliases 展开同义词

    def applies(self, value: Any) -> bool:
        """槽位有值就生效；配置了 trigger_values 时还要命中其中之一。"""
        if self.trigger_values:
            return str(value).strip() in self.trigger_values
        return bool(value)

    def terms_for(self, value: Any) -> Tuple[str, ...]:
        """固定词优先，否则把槽位值本身当作过滤词。"""
        text = str(value or "").strip()
        if self.terms:
            return self.terms
        return (text,) if text else ()

    def is_negative(self, value: Any) -> bool:
        """整槽排除，或当前值在 negative_values 里。"""
        return self.negative or str(value or "").strip() in self.negative_values


@dataclass(frozen=True)
class SlotVocabulary:
    """槽位抽取用的业务词表；理解层只按这里的配置取值。"""

    diet: Tuple[Tuple[str, Tuple[str, ...]], ...]       # (槽位值, 触发词)
    persona: Tuple[Tuple[str, Tuple[str, ...]], ...]
    quick_terms: Tuple[str, ...]
    quick_value: str
    spicy_aliases: Tuple[Tuple[str, Tuple[str, ...]], ...]  # LLM 辣度说法 -> 规范值
    spicy_text_terms: Tuple[Tuple[str, Tuple[str, ...]], ...]  # 原文抽取用的安全词
    taste_terms: Tuple[Tuple[str, Tuple[str, ...]], ...]     # 非辣口味（甜/酸/咸等）
    spicy_fallback: str
    constraint_terms: Tuple[str, ...]                   # 命中后把辣度改成 spicy_fallback
    category_terms: Tuple[str, ...]
    child_terms: Tuple[str, ...]
    add_fish_terms: Tuple[str, ...]                     # 任一命中
    add_fish_all_terms: Tuple[str, ...]                 # 必须同时命中
    minutes_pattern: str                                # “20分钟”里的数字
    servings_pattern: str                               # “8个人”里的数字
    exclude_pattern: str = ""                           # 从“不吃X”里抽排除项
    ingredient_pattern: str = ""                        # 从“带辣椒的菜”里抽食材


@dataclass(frozen=True)
class RouteThreshold:
    """一个路由键的置信度阈值；可按路由键单独校准。"""

    min_execute: float = 0.50           # 低于它先澄清，不猜
    retrieval_override: float = 0.70    # 达到它就覆盖 LLM 的“不检索”
    no_retrieval: float = 0.80          # 达到它的闲聊/域外意图可以拦截检索


# ---------------------------------------------------------------------------
# 规则兜底的核心词；ROUTE_SPECS 在此基础上加提示词/扩展词
# ---------------------------------------------------------------------------

INGREDIENT_PAIRING_TERMS: Tuple[str, ...] = (
    "一起吃", "同食", "相克", "搭配", "禁忌", "能不能一起", "为什么不能",
)
FOOD_SAFETY_TERMS: Tuple[str, ...] = (
    "过敏", "过敏原", "中毒", "食品安全", "沙门氏菌", "黄曲霉素",
    "寄生虫", "孕妇", "小孩能吃",
)
NUTRITION_VALUE_TERMS: Tuple[str, ...] = (
    "千卡", "大卡", "热量", "卡路里", "蛋白质", "脂肪", "多少卡",
)
NUTRITION_BENEFIT_TERMS: Tuple[str, ...] = (
    "营养", "功效", "好处", "适合", "滋补", "健康", "维生素",
)
RECOMMEND_SCENARIO_TERMS: Tuple[str, ...] = (
    "几个人", "人数", "聚餐", "荤素", "配菜", "加鱼", "一桌", "宴客",
)
RECIPE_INGREDIENTS_TERMS: Tuple[str, ...] = (
    "食材", "原料", "调料", "配料", "用料",
    "需要什么", "需要哪些", "准备什么", "必备",
)
RECOMMEND_LIST_TERMS: Tuple[str, ...] = (
    "推荐", "有什么菜", "哪些菜", "吃什么", "低热量",
    "减脂", "减肥", "新手", "快手", "简单", "家常",
)
INGREDIENT_PROPERTY_TERMS: Tuple[str, ...] = (
    "去腥", "焯水", "腌制", "凉拌", "微波炉", "空气炸锅", "高压锅",
    "挑选", "储存", "怎么切", "预处理",
)
RECIPE_OVERVIEW_TERMS: Tuple[str, ...] = (
    "菜系", "口味", "难度", "介绍", "是什么菜", "份量", "几个人吃", "特点",
)


# ---------------------------------------------------------------------------
# 6 个顶层意图 -> 12 个路由键
# ---------------------------------------------------------------------------

TOP_INTENTS: Tuple[str, ...] = (
    "recipe_detail",
    "ingredient_info",
    "recommendation",
    "nutrition_health",
    "chitchat",
    "out_of_scope",
)

INTENT_TO_KEYS: Dict[str, Tuple[str, ...]] = {
    "recipe_detail": ("recipe_ingredients", "recipe_steps_time", "recipe_overview"),
    "ingredient_info": ("ingredient_pairing", "ingredient_property"),
    "recommendation": ("recommend_list", "recommend_scenario"),
    "nutrition_health": ("nutrition_value", "nutrition_benefit", "food_safety"),
    "chitchat": ("chitchat",),
    "out_of_scope": ("out_of_scope",),
}

NO_RETRIEVAL_KEYS: Tuple[str, ...] = ("chitchat", "out_of_scope")


ROUTE_SPECS: Dict[str, RouteKeySpec] = {
    "recipe_ingredients": RouteKeySpec(
        key="recipe_ingredients",
        intent="recipe_detail",
        evidence_profile=("LIST",),
        engine="graph_entity",
        source="graph:REQUIRES 全量 + 菜谱原文",
        need_complete=True,
        default_top_k=20,
        validator="list",
        keywords=RECIPE_INGREDIENTS_TERMS,
        entity_first=True,
        expand_per_recipe=True,
        boost_with_entity=True,
    ),
    "recipe_steps_time": RouteKeySpec(
        key="recipe_steps_time",
        intent="recipe_detail",
        evidence_profile=("LIST", "VALUE", "TEXT"),
        engine="graph_entity",
        source="graph:CONTAINS_STEP + 菜谱原文操作 + tags",
        need_complete=True,
        default_top_k=30,
        validator="steps",
        keywords=("怎么做", "步骤", "几分钟", "多久", "火候", "蒸", "煮", "炒", "时间", "操作"),
        entity_first=True,
        expand_per_recipe=True,
        value_policy="exact_only",
        boost_with_entity=True,
    ),
    "recipe_overview": RouteKeySpec(
        key="recipe_overview",
        intent="recipe_detail",
        evidence_profile=("VALUE", "TEXT"),
        engine="graph_entity",
        source="graph:Recipe 属性 + 菜谱原文",
        need_complete=False,
        default_top_k=3,
        validator="overview",
        # 用户口径：“X 的做法是什么”算概览，不展开成步骤
        keywords=RECIPE_OVERVIEW_TERMS + ("做法",),
        entity_first=True,
        expand_per_recipe=True,
        boost_with_entity=True,
    ),
    "ingredient_pairing": RouteKeySpec(
        key="ingredient_pairing",
        intent="ingredient_info",
        evidence_profile=("REL", "TEXT"),
        engine="tips_pairing",
        source="tips:食材相克与禁忌.md（不走图）",
        need_complete=False,
        default_top_k=3,
        validator="pair",
        keywords=INGREDIENT_PAIRING_TERMS + ("食物相克",),
        reasoning_required=True,
        topic_mode="pair",
        disclaimer=True,
    ),
    "ingredient_property": RouteKeySpec(
        key="ingredient_property",
        intent="ingredient_info",
        evidence_profile=("TEXT",),
        engine="tips_technique",
        source="tips:learn/*.md + 菜谱原文",
        need_complete=False,
        default_top_k=3,
        validator="technique",
        keywords=INGREDIENT_PROPERTY_TERMS + ("处理",),
        entity_first=True,
        depends_on_previous=("ingredient_pairing",),
    ),
    "recommend_list": RouteKeySpec(
        key="recommend_list",
        intent="recommendation",
        evidence_profile=("FILTER",),
        engine="graph_entity",
        source="graph:Recipe 的 tags/name/description",
        need_complete=False,
        default_top_k=8,
        validator="recommend",
        keywords=RECOMMEND_LIST_TERMS + ("好吃的",),
        prefix_shared_recipe=False,
        depends_on_previous=("nutrition_value",),
    ),
    "recommend_scenario": RouteKeySpec(
        key="recommend_scenario",
        intent="recommendation",
        evidence_profile=("RULE", "FILTER"),
        engine="tips_rule",
        source="tips:如何选择现在吃什么.md + graph 分类候选",
        need_complete=False,
        default_top_k=10,
        validator="scenario",
        keywords=RECOMMEND_SCENARIO_TERMS + ("小孩",),
        prefix_shared_recipe=False,
        reasoning_required=True,
    ),
    "nutrition_value": RouteKeySpec(
        key="nutrition_value",
        intent="nutrition_health",
        evidence_profile=("VALUE",),
        engine="graph_entity",
        source="菜谱原文精确数值",
        need_complete=False,
        default_top_k=1,
        validator="value",
        keywords=NUTRITION_VALUE_TERMS + ("多少千卡",),
        entity_first=True,
        expand_per_recipe=True,
        reasoning_required=True,
        value_policy="exact_only",
        boost_with_entity=True,
    ),
    "nutrition_benefit": RouteKeySpec(
        key="nutrition_benefit",
        intent="nutrition_health",
        evidence_profile=("TEXT",),
        engine="graph_entity",
        source="菜谱原文描述 + 向量兜底",
        need_complete=False,
        default_top_k=5,
        validator="benefit",
        keywords=NUTRITION_BENEFIT_TERMS,
        entity_first=True,
        expand_per_recipe=True,
        depends_on_previous=("nutrition_value",),
        disclaimer=True,
        boost_with_entity=True,
    ),
    "food_safety": RouteKeySpec(
        key="food_safety",
        intent="nutrition_health",
        evidence_profile=("LIST", "TEXT"),
        engine="tips_safety",
        source="tips:learn/食品安全.md",
        need_complete=True,
        default_top_k=3,
        validator="safety",
        keywords=FOOD_SAFETY_TERMS + (
            "安全", "没熟", "熟透", "中心温度", "变质", "发芽", "发霉", "能吃吗",
        ),
        disclaimer=True,
    ),
    "chitchat": RouteKeySpec(
        key="chitchat",
        intent="chitchat",
        evidence_profile=("NONE",),
        engine="none",
        source="none",
        need_complete=False,
        default_top_k=0,
        validator="",
        keywords=("好吃", "喜欢", "觉得", "想吃", "好想", "谢谢", "哈哈", "你好", "在吗", "你自己", "会做饭", "累", "开心", "难过"),
        topic_mode="chitchat",
    ),
    "out_of_scope": RouteKeySpec(
        key="out_of_scope",
        intent="out_of_scope",
        evidence_profile=("NONE",),
        engine="none",
        source="none",
        need_complete=False,
        default_top_k=0,
        validator="",
        keywords=("天气", "电影", "微信", "股票", "新闻", "翻译", "写代码", "编程", "手机", "电脑", "旅游", "机票", "酒店", "快递", "游戏", "音乐", "视频", "政治", "数学", "作业"),
        refuse=True,
    ),
}


# “X 的做法是什么”按用户口径算概览；带明确步骤/时间词的仍算步骤
OVERVIEW_WORDING_TERMS: Tuple[str, ...] = ("做法",)
OVERVIEW_EXCLUDE_TERMS: Tuple[str, ...] = (
    "步骤", "怎么做", "怎么煮", "怎么炒", "怎么蒸",
    "多久", "几分钟", "火候", "时间",
)

# 规则兜底：顺序就是优先级，第一条命中即返回。
# 业务词表全部放在这里，rules.py 只负责按表匹配。
RULE_SPECS: Tuple[RuleSpec, ...] = (
    # 域外：没有明确饮食线索时，天气/电影/微信等优先判域外
    RuleSpec(
        route_key="out_of_scope",
        terms=ROUTE_SPECS["out_of_scope"].keywords,
        skip_if_any=(
            "吃", "喝", "菜", "饭", "面", "汤", "肉", "鱼", "蛋",
            "食材", "营养", "厨房", "烹饪", "食谱", "调料",
        ),
    ),
    # 闲聊：主观比较 / 情绪 / 寒暄
    RuleSpec(
        route_key="chitchat",
        terms=("哪个更好吃", "哪个好吃", "你更喜欢", "你觉得", "更好吃"),
        slots={"chitchat_type": "subjective"},
    ),
    RuleSpec(
        route_key="chitchat",
        terms=("好想", "想吃点", "馋", "突然想", "累", "开心", "难过", "哈哈"),
        slots={"chitchat_type": "emotion"},
        skip_if_request=True,
    ),
    RuleSpec(
        route_key="chitchat",
        terms=("你好", "谢谢", "在吗", "你自己", "会做饭吗", "你是谁"),
        slots={"chitchat_type": "social"},
    ),
    # 纯约束声明（“我不吃辣”）先记到状态里，不触发检索
    RuleSpec(
        route_key="chitchat",
        terms=("不吃辣", "不能吃辣", "忌辣", "忌口"),
        slots={"chitchat_type": "constraint"},
        skip_if_any=("推荐", "有什么", "哪些", "吃什么", "怎么", "做"),
    ),
    # 相克/同食：先于食品安全，因为“能不能一起吃”更具体
    RuleSpec(
        route_key="ingredient_pairing",
        terms=INGREDIENT_PAIRING_TERMS,
    ),
    RuleSpec(
        route_key="food_safety",
        terms=FOOD_SAFETY_TERMS,
    ),
    # 明确“要推荐一组菜”优先于营养数值（“低热量”里也含“热量”）
    RuleSpec(
        route_key="recommend_list",
        terms=(
            "推荐", "有什么菜", "哪些菜", "吃什么", "好吃的",
            "有什么好吃", "几道", "来几道", "帮我选",
        ),
    ),
    RuleSpec(route_key="recommend_list", terms=("菜",), require_request=True),
    RuleSpec(
        route_key="nutrition_value",
        terms=NUTRITION_VALUE_TERMS,
    ),
    RuleSpec(
        route_key="nutrition_benefit",
        terms=NUTRITION_BENEFIT_TERMS,
    ),
    RuleSpec(
        route_key="recommend_scenario",
        terms=RECOMMEND_SCENARIO_TERMS,
    ),
    RuleSpec(
        route_key="recipe_ingredients",
        terms=RECIPE_INGREDIENTS_TERMS,
    ),
    # 步骤/时间；“做法”单独处理，避免把“做法是什么”误判成步骤
    RuleSpec(
        route_key="recipe_steps_time",
        terms=(
            "怎么做", "步骤", "几分钟", "多久", "火候", "操作",
            "怎么蒸", "怎么煮", "怎么炒", "蒸多久", "煮多久",
        ),
    ),
    # 用户口径：“可乐鸡翅的做法是什么？”归 recipe_overview
    RuleSpec(route_key="recipe_overview", terms=OVERVIEW_WORDING_TERMS),
    # 推荐列表：命中 request_terms，或出现“低热量/减脂/新手”等词
    RuleSpec(route_key="recommend_list", require_request=True),
    RuleSpec(
        route_key="recommend_list",
        terms=RECOMMEND_LIST_TERMS,
    ),
    # 食材预处理/厨房技法
    RuleSpec(
        route_key="ingredient_property",
        terms=INGREDIENT_PROPERTY_TERMS,
    ),
    # 菜谱概览
    RuleSpec(
        route_key="recipe_overview",
        terms=RECIPE_OVERVIEW_TERMS,
    ),
)

# 追问分类：仅在已有 active_recipe 且出现指代/省略词时使用。
# 按顺序匹配，第一条命中即返回；最后一条不写 terms，作为兜底。
FOLLOW_UP_INGREDIENT_TERMS: Tuple[str, ...] = (
    "盐", "糖", "油", "水", "醋", "生抽", "老抽", "料酒", "淀粉",
)
FOLLOW_UP_RULES: Tuple[RuleSpec, ...] = (
    RuleSpec(route_key="recipe_ingredients", terms=("食材", "原料", "调料", "配料", "用料")),
    RuleSpec(route_key="recipe_steps_time", terms=("怎么做", "步骤", "怎么蒸", "怎么煮", "怎么炒")),
    RuleSpec(route_key="nutrition_value", terms=("千卡", "大卡", "热量", "卡路里")),
    RuleSpec(route_key="nutrition_benefit", terms=("营养", "功效", "好处")),
    # 问用量时，把具体调料名写进 ingredient 槽位
    RuleSpec(
        route_key="recipe_ingredients",
        terms=("多少盐", "多少糖", "多少油", "多少克", "毫升", "盐", "糖", "油"),
        slot_from_terms={"ingredient": FOLLOW_UP_INGREDIENT_TERMS},
    ),
    RuleSpec(route_key="recipe_steps_time", terms=("几分钟", "多久", "火候", "时间")),
    RuleSpec(
        route_key="ingredient_property",
        terms=("换成", "替代", "没有", "去腥", "焯水", "腌制", "凉拌"),
    ),
    # 兜底：“这道菜的做法是什么”按用户口径归概览
    RuleSpec(route_key="recipe_overview"),
)

# LLM 已给出 route_key 时，按措辞做最后归一化
WORDING_RULES: Tuple[WordingRule, ...] = (
    WordingRule(
        source_keys=("recipe_steps_time",),
        target_key="recipe_overview",
        terms=OVERVIEW_WORDING_TERMS,
        exclude_terms=OVERVIEW_EXCLUDE_TERMS,
    ),
)


# 每个路由键的置信度阈值；默认值保持原全局常量的行为，校准后只改这里。
DEFAULT_THRESHOLD = RouteThreshold()
ROUTE_THRESHOLDS: Dict[str, RouteThreshold] = {
    "default": DEFAULT_THRESHOLD,
    **{key: DEFAULT_THRESHOLD for key in ROUTE_SPECS},
}


def _load_thresholds(path: str) -> Dict[str, RouteThreshold]:
    """从 JSON 覆盖阈值；canary 用 ROUTE_THRESHOLDS_PATH 指向候选配置。"""
    if not path:
        return ROUTE_THRESHOLDS

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("加载路由阈值失败，使用默认值 %s: %s", path, exc)
        return ROUTE_THRESHOLDS

    merged = dict(ROUTE_THRESHOLDS)
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        merged[key] = RouteThreshold(
            min_execute=float(value.get("min_execute", DEFAULT_THRESHOLD.min_execute)),
            retrieval_override=float(
                value.get("retrieval_override", DEFAULT_THRESHOLD.retrieval_override)
            ),
            no_retrieval=float(
                value.get("no_retrieval", DEFAULT_THRESHOLD.no_retrieval)
            ),
        )
    return merged


ROUTE_THRESHOLDS = _load_thresholds(os.getenv("ROUTE_THRESHOLDS_PATH", ""))


# ---------------------------------------------------------------------------
# 业务词表
# ---------------------------------------------------------------------------

# 用户口语 -> 图谱/文档里的标准词。顺序无所谓，查的时候会全部展开。
ALIASES: Dict[str, Tuple[str, ...]] = {
    "豆腐": ("豆腐", "内脂豆腐", "嫩豆腐", "北豆腐"),
    "牛奶": ("牛奶", "全脂牛奶"),
    "鸡肉": ("鸡肉", "鸡胸肉", "鸡胸脯肉", "鸡腿肉", "鸡翅中"),
    "西红柿": ("西红柿", "番茄"),
    "土豆": ("土豆", "马铃薯"),
    "红薯": ("红薯", "番薯"),
    "芸豆": ("芸豆", "四季豆"),
    "豇豆": ("豇豆", "豆角"),
    "螃蟹": ("螃蟹", "蟹", "蟹类"),
    "虾": ("虾", "虾类", "虾蟹"),
    "豆浆": ("豆浆",),
    "柿子": ("柿子",),
    "西瓜": ("西瓜",),
}

# 规则兜底用的主题词；也用于推荐过滤。
TOPIC_TERMS: Dict[str, Tuple[str, ...]] = {
    "diet": ("减脂", "低热量", "低卡", "减肥", "健康餐"),
    "quick": ("快手", "简单", "快速", "10分钟", "十分钟"),
    "beginner": ("新手", "零基础", "简单"),
    "child": ("小孩", "儿童", "孩子", "宝宝"),
    "non_spicy": ("不辣", "微辣", "清淡"),
    "spicy": ("辣", "麻辣", "香辣", "特辣"),
    "fish": ("鱼", "鱼类", "水产"),
    "meal_plan": ("聚餐", "人数", "几个人", "荤素", "配菜", "加鱼", "一桌", "宴客"),
}

# 菜系词；如果图谱里新增菜系，这里补一行即可。
CUISINE_TERMS: Tuple[str, ...] = (
    "川菜", "粤菜", "鲁菜", "湘菜", "苏菜", "浙菜", "闽菜", "徽菜",
    "东北菜", "西北菜", "泰国菜", "日本菜", "韩式", "西餐", "家常菜",
)

# 中文里表示“指代/省略”的词，用于判断是否需要改写。
REFERENCE_TERMS: Tuple[str, ...] = (
    "它", "这个", "那个", "这道", "那这道", "这菜", "那菜",
    "还要", "再来", "换成", "替代", "多少盐", "多少糖", "多少油",
    "多少克", "几分钟", "多久", "火候", "然后呢", "继续",
)

# 领域内提示词：规则兜底时，避免把纯饮食问题误判成域外。
FOOD_HINTS: Tuple[str, ...] = (
    "吃", "菜", "饭", "面", "汤", "肉", "鱼", "蛋", "蔬", "营养",
    "厨房", "烹饪", "食材", "做", "煮", "炒", "蒸", "烤", "炸", "调料",
)

# 明确表示“要一组菜/要建议”的词，避免把“好想吃”误判成推荐。
REQUEST_TERMS: Tuple[str, ...] = (
    "推荐", "有什么", "哪些", "吃什么", "做什么", "来几道", "给几道", "帮我选",
)

# 图谱里的占位/泛化节点，不能当实体。
GENERIC_ENTITY_NAMES: Tuple[str, ...] = ("菜谱", "食材", "菜品", "食物")

# 图谱 Recipe.category 的真实分类；槽位词表用这一份。
CATEGORY_NAMES: Tuple[str, ...] = (
    "素菜", "荤菜", "汤类", "主食", "甜品", "水产", "早餐", "凉菜", "饮料",
)

# LLM 可能把菜类写进 diet（“素菜”），也可能把食材类别写成“蔬菜”；
# 统一到图谱的 Recipe.category。
CATEGORY_ALIASES: Dict[str, str] = {
    "素菜": "素菜", "素": "素菜", "素食": "素菜", "蔬菜": "素菜", "青菜": "素菜",
    "荤菜": "荤菜", "肉菜": "荤菜", "肉类": "荤菜", "肉": "荤菜",
    "汤类": "汤类", "汤": "汤类",
    "主食": "主食", "饭": "主食", "面": "主食", "粉": "主食",
    "甜品": "甜品", "甜点": "甜品", "蛋糕": "甜品",
    "水产": "水产", "海鲜": "水产", "鱼": "水产",
    "早餐": "早餐", "饮料": "饮料", "饮品": "饮料", "半成品": "半成品",
    "凉菜": "凉菜",
}

# 减脂/健康目标词；和菜类区分，不当作 category 过滤。
DIET_TERMS: Tuple[str, ...] = ("减脂", "减肥", "低热量", "低卡", "健康")

CHILD_TERMS: Tuple[str, ...] = ("儿童", "小孩", "孩子", "宝宝")

# “不吃辣”类约束；“忌口/不喜欢”等更宽的否定词只在纯约束声明里用。
SPICY_CONSTRAINT_TERMS: Tuple[str, ...] = (
    "不吃辣", "不能吃辣", "不喜欢吃辣", "不爱吃辣", "不想吃辣", "忌辣",
)

# 槽位抽取：所有业务词都在这里，理解层只负责按槽位写入。
SLOT_VOCABULARY = SlotVocabulary(
    diet=(
        ("减脂", ("减脂", "减肥")),
        ("低热量", ("低热量", "低卡")),
    ),
    persona=(
        ("新手", ("新手", "零基础")),
        ("儿童", CHILD_TERMS),
        ("老人", ("老人", "长辈")),
    ),
    quick_terms=("快手", "快速"),
    quick_value="快手",
    spicy_aliases=(
        ("不辣", ("不辣", "清淡")),
        ("微辣", ("微辣", "小辣", "轻辣")),
        ("中辣", ("中辣", "中等辣")),
        ("特辣", ("特辣", "变态辣", "重辣", "超辣")),
        ("辣", ("辣", "麻辣", "香辣", "酸辣", "甜辣", "辣味", "辣口", "辣的")),
    ),
    # 原句抽取不用“麻辣/香辣”，避免把“麻辣香锅”这种菜名当成口味偏好
    spicy_text_terms=(
        ("不辣", ("不辣", "清淡")),
        ("微辣", ("微辣", "小辣")),
        ("中辣", ("中辣",)),
        ("特辣", ("特辣", "变态辣", "重辣", "超辣")),
        (
            "辣",
            ("辣口", "辣味", "辣的", "麻辣的", "香辣的", "酸辣的", "甜辣的",
             "辣一点", "辣一些", "辣点"),
        ),
    ),
    taste_terms=(
        ("甜", ("甜味", "甜的", "甜口", "甜食")),
        ("酸", ("酸味", "酸的", "酸口")),
        ("咸", ("咸味", "咸的", "咸口")),
        ("苦", ("苦味", "苦的")),
        ("鲜", ("鲜味", "鲜的")),
    ),
    spicy_fallback="不辣",
    constraint_terms=SPICY_CONSTRAINT_TERMS,
    category_terms=CATEGORY_NAMES,
    child_terms=CHILD_TERMS,
    add_fish_terms=("加鱼",),
    add_fish_all_terms=("鱼", "荤"),
    minutes_pattern=r"(\d+)\s*分钟",
    servings_pattern=r"(\d+)\s*(?:个)?人",
    # “不喜欢香菜 / 不要甜品 / 不放辣椒”都抽出排除项
    exclude_pattern=(
        r"(?:不喜欢吃|不喜欢|不爱吃|不爱|不想吃|不能吃|不吃|不要|"
        r"别吃|别放|不放|不加|不含|受不了)(?:吃|放|加|用|含)?([一-龥]{1,6})"
    ),
    # “带辣椒的菜 / 含牛肉的菜”；“带鱼的做法”不会命中
    ingredient_pattern=(
        r"(?<!不)(?:带有|带了|带|含有|含|放了|放|加了|加)"
        r"([一-龥]{1,8})的(?=菜|[，。？！,.?!]|$)"
    ),
)

# 推荐候选过滤：新增一个维度时只在这里加一行，evidence 会生成对应 Cypher。
# fields 可取 name / tags / category / cuisine / ingredient；
# ingredient 会走 (:Recipe)-[:REQUIRES]->(:Ingredient) 关系。
CANDIDATE_FILTERS: Tuple[CandidateFilterSpec, ...] = (
    CandidateFilterSpec(slot="cuisine", fields=("cuisine",)),
    CandidateFilterSpec(slot="category", fields=("category",)),
    CandidateFilterSpec(
        slot="diet", fields=("name", "tags"), terms=("减脂", "低热量", "减肥"),
    ),
    CandidateFilterSpec(
        slot="time", fields=("tags",),
        terms=("快手", "简单", "10分钟"), trigger_values=("快手", "快速"),
    ),
    CandidateFilterSpec(
        slot="max_minutes", fields=("tags",), terms=("快手", "简单", "10分钟"),
    ),
    CandidateFilterSpec(
        slot="persona", fields=("tags",),
        terms=("新手友好", "简单"), trigger_values=("新手",),
    ),
    CandidateFilterSpec(
        slot="persona", fields=("tags", "category"),
        terms=("甜",), trigger_values=("儿童",),
    ),
    CandidateFilterSpec(
        slot="has_child", fields=("tags", "category"), terms=("甜",),
    ),
    # 正向按“辣”找菜，不辣则把名称/标签/食材带辣的都排除
    CandidateFilterSpec(
        slot="spicy",
        fields=("name", "tags", "ingredient"),
        terms=("辣",),
        trigger_values=("不辣", "微辣", "中辣", "特辣", "辣"),
        negative_values=("不辣",),
    ),
    CandidateFilterSpec(
        slot="ingredient",
        fields=("ingredient", "name", "tags"),
        multi_value=True,
        expand_aliases=True,
    ),
    # “不喜欢香菜”这类否定偏好，名称/分类/标签/食材任一命中都排除
    CandidateFilterSpec(
        slot="exclude",
        fields=("name", "category", "cuisine", "tags", "ingredient"),
        negative=True,
        multi_value=True,
        expand_aliases=True,
    ),
    # 非辣口味（甜/酸/咸等）也按同一套机制过滤
    CandidateFilterSpec(slot="flavor", fields=("name", "tags", "ingredient")),
    CandidateFilterSpec(slot="taste", fields=("name", "tags", "ingredient")),
)

# 纯约束声明：不检索，只把约束写进会话状态。
PURE_CONSTRAINT = ConstraintSpec(
    terms=SPICY_CONSTRAINT_TERMS + (
        "忌口", "不喜欢", "不爱吃", "不想吃", "不要", "别放", "不放", "不加", "不含", "受不了",
    ),
    # 不能只写“菜”：香菜/白菜/川菜里的“菜”不是推荐请求
    request_terms=(
        "推荐", "有什么", "哪些", "吃什么", "怎么", "做",
        "来几道", "给几道", "帮我选",
    ),
    route_key="chitchat",
    slots={"chitchat_type": "constraint"},
)


# ---------------------------------------------------------------------------
# 给 LLM 看的业务 Schema（不是建表语句，是理解层需要知道的字段/关系）
# ---------------------------------------------------------------------------

NEO4J_SCHEMA = """
Neo4j 图谱（CookWise）：
- (:Recipe {nodeId, name, category, cuisineType, difficulty, prepTime, cookTime, servings, tags, filePath})
- (:Ingredient {nodeId, name, category, description})
- (:CookingStep {nodeId, name, description, stepNumber, timeEstimate})
- (:Category {name})
- (:Recipe)-[:REQUIRES {amount, unit}]->(:Ingredient)
- (:Recipe)-[:CONTAINS_STEP {stepOrder}]->(:CookingStep)
- (:Recipe)-[:BELONGS_TO_CATEGORY]->(:Category)
- (:Recipe)-[:SIMILAR]->(:Recipe)
相克/禁忌关系不在图谱里，只在 data/tips/食材相克与禁忌.md。
"""

MILVUS_SCHEMA = """
Milvus 集合 cooking_knowledge（1138 条菜谱 chunk，BGE-small-zh 512 维，COSINE）：
- 字段：id, vector, text, node_id, recipe_name, node_type, category,
        cuisine_type, difficulty, doc_type, chunk_id, parent_id
- 当前只有 doc_type=chunk、node_type=Recipe 的菜谱块；tips 文档尚未入索引。
"""

TIPS_SOURCES = """
- 相克/同食：data/tips/食材相克与禁忌.md
- 食品安全：data/tips/learn/食品安全.md
- 人数/荤素/加鱼：data/tips/如何选择现在吃什么.md
- 技法（去腥/焯水/腌制/蒸/煮等）：data/tips/learn/*.md
"""


@dataclass(frozen=True)
class DomainConfig:
    """一个领域的完整路由配置。换领域时替换这个对象即可。"""

    name: str
    top_intents: Tuple[str, ...]
    route_specs: Dict[str, RouteKeySpec]
    rule_specs: Tuple[RuleSpec, ...]
    follow_up_rules: Tuple[RuleSpec, ...]
    wording_rules: Tuple[WordingRule, ...]
    intent_to_keys: Dict[str, Tuple[str, ...]]
    no_retrieval_keys: Tuple[str, ...]
    aliases: Dict[str, Tuple[str, ...]]
    topic_terms: Dict[str, Tuple[str, ...]]
    cuisine_terms: Tuple[str, ...]
    reference_terms: Tuple[str, ...]
    food_hints: Tuple[str, ...]
    request_terms: Tuple[str, ...]
    category_aliases: Dict[str, str]
    diet_terms: Tuple[str, ...]
    generic_entity_names: Tuple[str, ...]
    slot_vocabulary: SlotVocabulary
    candidate_filters: Tuple[CandidateFilterSpec, ...]
    constraint_spec: Optional[ConstraintSpec]
    clarify_messages: Dict[str, str]
    default_route_key: str
    out_of_scope_key: str
    empty_query_key: str
    neo4j_schema: str
    milvus_schema: str
    tips_sources: str
    persisted_slots: Tuple[str, ...] = ()   # 跨轮延续到会话状态的槽位
    route_thresholds: Dict[str, RouteThreshold] = field(default_factory=dict)

    def spec(self, route_key: str) -> RouteKeySpec:
        """按路由键取定义；未知键回退到领域配置的默认路由。"""
        return self.route_specs.get(
            route_key, self.route_specs[self.default_route_key]
        )

    def threshold(self, route_key: str) -> RouteThreshold:
        """按路由键取阈值；没有单独配置就用 default。"""
        return (
            self.route_thresholds.get(route_key)
            or self.route_thresholds.get("default")
            or DEFAULT_THRESHOLD
        )

    def route_table_text(self) -> str:
        """生成 prompt 里的路由键表。"""
        lines = []
        for spec in self.route_specs.values():
            lines.append(
                f"- {spec.key}（{spec.intent}，证据画像 {'/'.join(spec.evidence_profile)}）："
                f"{spec.source}；例：{'、'.join(spec.keywords[:4])}"
            )
        return "\n".join(lines)


DOMAIN = DomainConfig(
    name="cookwise",
    top_intents=TOP_INTENTS,
    route_specs=ROUTE_SPECS,
    rule_specs=RULE_SPECS,
    follow_up_rules=FOLLOW_UP_RULES,
    wording_rules=WORDING_RULES,
    intent_to_keys=INTENT_TO_KEYS,
    no_retrieval_keys=NO_RETRIEVAL_KEYS,
    aliases=ALIASES,
    topic_terms=TOPIC_TERMS,
    cuisine_terms=CUISINE_TERMS,
    reference_terms=REFERENCE_TERMS,
    food_hints=FOOD_HINTS,
    request_terms=REQUEST_TERMS,
    category_aliases=CATEGORY_ALIASES,
    diet_terms=DIET_TERMS,
    generic_entity_names=GENERIC_ENTITY_NAMES,
    slot_vocabulary=SLOT_VOCABULARY,
    candidate_filters=CANDIDATE_FILTERS,
    constraint_spec=PURE_CONSTRAINT,
    clarify_messages={
        "no_intent": "请再说明一下你想了解哪道菜或哪种食材。",
        "ambiguous": "请再说明一下你指的是哪道菜或哪种食材。",
        "low_confidence": "我不太确定你问的是哪道菜或哪种食材，可以补充一下吗？",
        "reference_missing": "你指的是哪道菜？请补充菜名，我再帮你查。",
        "empty_query": "请告诉我你想了解哪道菜或哪种食材。",
    },
    default_route_key="recipe_overview",
    out_of_scope_key="out_of_scope",
    empty_query_key="chitchat",
    neo4j_schema=NEO4J_SCHEMA,
    milvus_schema=MILVUS_SCHEMA,
    tips_sources=TIPS_SOURCES,
    persisted_slots=("diet", "persona", "servings", "spicy", "exclude", "cuisine"),
    route_thresholds=ROUTE_THRESHOLDS,
)
