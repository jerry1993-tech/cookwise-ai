#!/usr/bin/env python3
"""
从项目真实数据合成黄金集（600 条）。

数据来源：
- Neo4j：323 道菜谱、食材、步骤、分类/菜系/标签
- data/dishes/*.md：菜谱简介、热量原文
- data/tips/*.md：相克、食品安全、技法

输出：
- tests/gold/golden_set.jsonl
- tests/gold/entities.json（校验用的离线实体快照）

用法：
    python scripts/eval/generate_gold.py
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from neo4j import GraphDatabase

from config import DEFAULT_CONFIG
from rag_modules.query_routing.domain import (
    GENERIC_ENTITY_NAMES,
    NO_RETRIEVAL_KEYS,
    ROUTE_SPECS,
)
from rag_modules.query_routing.tips_knowledge import TipsKnowledgeBase

IN_DOMAIN_KEYS = [key for key in ROUTE_SPECS if key not in NO_RETRIEVAL_KEYS]
ALL_KEYS = list(ROUTE_SPECS)
KCAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(千卡|大卡|kcal|卡路里)", re.IGNORECASE)

BENEFIT_TERMS = (
    "高蛋白", "优质蛋白", "蛋白质", "低脂肪", "低脂", "脂肪",
    "膳食纤维", "维生素", "矿物质", "碳水", "营养", "健康",
    "滋补", "开胃", "易消化", "美容", "免疫力", "补钙", "补铁",
)

SAFETY_CASES: List[Tuple[str, str]] = [
    ("四季豆没熟会中毒吗？", "中毒"),
    ("生豆浆能直接喝吗？", "中毒"),
    ("发芽的土豆还能吃吗？", "中毒"),
    ("野生蘑菇为什么不能随便吃？", "中毒"),
    ("木耳泡太久会有毒吗？", "中毒"),
    ("鲜黄花菜怎么处理才安全？", "中毒"),
    ("常见的食物过敏原有哪些？", "过敏"),
    ("儿童食物过敏要注意什么？", "过敏"),
    ("成人食物过敏有哪些表现？", "过敏"),
    ("鸡蛋有沙门氏菌风险吗？", "沙门氏菌感染"),
    ("沙门氏菌要怎么预防？", "沙门氏菌感染"),
    ("发霉的花生还能吃吗？", "黄曲霉素"),
    ("怎么避免黄曲霉素？", "黄曲霉素"),
    ("红心甘蔗能吃吗？", "3-硝基丙酸"),
    ("甘蔗中毒是怎么回事？", "3-硝基丙酸"),
    ("椰子放久了会有毒吗？", "3-硝基丙酸"),
    ("生鱼片有寄生虫风险吗？", "寄生虫"),
    ("田螺里的寄生虫怎么预防？", "寄生虫"),
    ("黄鳝有寄生虫吗？", "寄生虫"),
    ("牛蛙没熟能吃吗？", "寄生虫"),
    ("怎么判断食物已经熟透了？", "食品安全温度"),
    ("肉的中心温度要多少度？", "食品安全温度"),
    ("剩菜要加热到什么温度？", "食品安全温度"),
]

CHITCHAT_CASES = [
    "你好", "谢谢你", "今天好累", "我心情不好", "你是谁", "你能做什么",
    "哈哈", "早上好", "晚安", "我最近在学做饭", "做饭好难啊",
    "你觉得做饭有意思吗", "我刚开始学做菜", "今天不想做饭",
    "做饭让我很放松", "你吃饭了吗", "周末愉快", "我有点饿了",
    "厨房好乱", "我买了一个新锅", "今天加班到现在", "有点困",
    "我喜欢吃家常菜", "你觉得我适合学做饭吗", "辛苦了",
    "我先去忙了", "回头聊", "你真好", "今天过得怎么样", "在吗",
]

OOS_CASES = [
    "今天天气怎么样？", "帮我写一段Python代码", "股票明天会涨吗？",
    "帮我订一张机票", "1+1等于几？", "明天会下雨吗？",
    "帮我翻译这句话", "世界上最高的山是哪座？", "iPhone的截图快捷键是什么？",
    "护照办理流程是什么？", "讲一个睡前故事", "提高工作效率的方法有哪些？",
    "足球比赛几点开始？", "帮我算一下房贷", "吉他的英文怎么说？",
    "今天油价涨了吗？", "自行车链条掉了，能修吗？", "巴黎是哪个国家的首都？",
    "笔记本电脑的内存越大越好吗？", "最近发生了什么大事？", "帮我写一封邮件",
    "睡眠不足有什么危害？", "汽车保养一次贵吗？", "跑步前要拉伸吗？",
    "这款手机的屏幕多大？", "明天是星期几？", "地铁末班车几点？",
    "去火车站坐几号线？", "这部电影的导演是谁？", "附近有银行吗？",
]

# 闲聊 + 任务的多意图边界：用情绪表达，避免“你好”被模型当成可忽略的寒暄
BOUNDARY_CHITCHAT = [
    "今天工作好累", "我心情有点低落", "最近压力有点大",
    "今天被老板批评了", "我有点焦虑", "昨晚没睡好",
]

# 食材 → 技法要符合常识，避免“酵母怎么凉拌”这类伪正例
PROPERTY_CASES = [
    ("巴沙鱼怎么去腥？", "去腥"),
    ("鸡胸肉如何去腥？", "去腥"),
    ("羊肉怎么去腥？", "去腥"),
    ("菠菜怎么焯水？", "焯水"),
    ("西兰花焯水要注意什么？", "焯水"),
    ("排骨怎么焯水？", "焯水"),
    ("鸡翅怎么腌制？", "腌制"),
    ("牛肉怎么腌制入味？", "腌制"),
    ("羊肉腌有什么技巧？", "腌制"),
    ("黄瓜怎么凉拌？", "凉拌"),
    ("木耳凉拌有什么技巧？", "凉拌"),
    ("菠菜凉拌前要处理吗？", "凉拌"),
    ("牛奶能用微波炉加热吗？", "微波炉"),
    ("米饭用微波炉怎么加热？", "微波炉"),
    ("馒头微波炉怎么热？", "微波炉"),
    ("鸡翅能用空气炸锅做吗？", "空气炸锅"),
    ("土豆能用空气炸锅做吗？", "空气炸锅"),
    ("茄子用空气炸锅要注意什么？", "空气炸锅"),
    ("排骨能用高压锅炖吗？", "高压锅"),
    ("牛肉能用高压锅炖吗？", "高压锅"),
    ("黄豆用高压锅要注意什么？", "高压锅"),
]

CLARIFY_CASES = [
    "这道菜需要哪些食材？", "它要煮多久？", "这个怎么做？",
    "那个热量多少？", "它的营养怎么样？", "这道菜难不难做？",
    "这个是什么菜？", "它适合几个人吃？", "那个要准备什么？",
    "帮我看看这个", "它的做法是什么？", "这个和什么不能一起吃？",
    "它有什么功效？", "这道菜是哪里的菜？", "这个要烤多久？",
    "那个需要哪些调料？", "它为什么不能一起吃？", "这道菜怎么保存？",
    "它的热量高吗？", "这个菜辣不辣？", "那个的步骤是什么？",
    "这道菜用不用焯水？", "它的口感怎么样？", "这个要蒸几分钟？",
    "那个难学吗？", "它需要什么锅？", "这道菜甜不甜？",
    "那个是荤菜吗？", "它的成本高吗？", "这个适合新手吗？",
]


@dataclass
class Recipe:
    name: str
    category: str
    cuisine: str
    difficulty: Optional[float]
    prep: str
    cook: str
    servings: str
    tags: List[str]
    file_path: str
    ingredients: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    markdown: str = ""
    intro: str = ""
    kcal: List[Tuple[str, str]] = field(default_factory=list)
    benefit_term: str = ""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _load_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _intro_from_markdown(markdown: str) -> str:
    """取一级标题之后、第一个二级标题之前的简介。"""
    head = markdown.split("\n## ", 1)[0]
    lines = [line.strip() for line in head.splitlines() if line.strip()]
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return _collapse(" ".join(lines))


def _benefit_term(intro: str) -> str:
    for term in BENEFIT_TERMS:
        if term in intro:
            return term
    return ""


def load_recipes(driver) -> List[Recipe]:
    """从 Neo4j 和 Markdown 汇总菜谱信息。"""
    recipes: Dict[str, Recipe] = {}

    with driver.session() as session:
        rows = session.run(
            """
            MATCH (r:Recipe)
            WHERE r.name IS NOT NULL AND r.filePath IS NOT NULL
            RETURN r.name AS name, r.category AS category,
                   r.cuisineType AS cuisine, r.difficulty AS difficulty,
                   r.prepTime AS prep, r.cookTime AS cook, r.servings AS servings,
                   r.tags AS tags, r.filePath AS file_path
            ORDER BY r.name
            """
        )
        for row in rows:
            name = _text(row["name"])
            if not name or name in GENERIC_ENTITY_NAMES:
                continue
            recipes[name] = Recipe(
                name=name,
                category=_text(row["category"]),
                cuisine=_text(row["cuisine"]),
                difficulty=row["difficulty"],
                prep=_text(row["prep"]),
                cook=_text(row["cook"]),
                servings=_text(row["servings"]),
                tags=[part.strip() for part in _text(row["tags"]).split(",") if part.strip()],
                file_path=_text(row["file_path"]),
            )

        rows = session.run(
            """
            MATCH (r:Recipe)-[req:REQUIRES]->(i:Ingredient)
            WHERE r.name IS NOT NULL
            RETURN r.name AS recipe, i.name AS name, req.isMain AS is_main,
                   req.amount AS amount, req.unit AS unit, i.category AS category
            ORDER BY r.name, is_main DESC, i.name
            """
        )
        for row in rows:
            recipe = recipes.get(_text(row["recipe"]))
            if recipe is None:
                continue
            recipe.ingredients.append({
                "name": _text(row["name"]),
                "is_main": bool(row["is_main"]),
                "amount": _text(row["amount"]),
                "unit": _text(row["unit"]),
                "category": _text(row["category"]),
            })

        rows = session.run(
            """
            MATCH (r:Recipe)-[c:CONTAINS_STEP]->(s:CookingStep)
            WHERE r.name IS NOT NULL
            RETURN r.name AS recipe, s.description AS description,
                   s.timeEstimate AS time_estimate, s.stepNumber AS step_number
            ORDER BY r.name, coalesce(c.stepOrder, s.stepNumber, 999)
            """
        )
        for row in rows:
            recipe = recipes.get(_text(row["recipe"]))
            if recipe is None:
                continue
            recipe.steps.append({
                "description": _text(row["description"]),
                "time": _text(row["time_estimate"]),
                "number": row["step_number"],
            })

    for recipe in recipes.values():
        path = ROOT / "data" / recipe.file_path.replace("\\", "/")
        recipe.markdown = _load_markdown(path)
        recipe.intro = _intro_from_markdown(recipe.markdown)
        recipe.benefit_term = _benefit_term(recipe.intro)
        recipe.kcal = [
            (f"{match.group(1)} {match.group(2)}", match.group(2))
            for match in KCAL_RE.finditer(recipe.markdown)
        ]

    return sorted(recipes.values(), key=lambda item: item.name)


class Cursor:
    """按顺序循环取实体，保证生成结果可复现。"""

    def __init__(self, items: List[Any]):
        if not items:
            raise ValueError("候选实体不能为空")
        self.items = items
        self.index = 0

    def next(self) -> Any:
        item = self.items[self.index % len(self.items)]
        self.index += 1
        return item


class GoldBuilder:
    def __init__(self, recipes: List[Recipe], tips: TipsKnowledgeBase):
        self.recipes = recipes
        self.tips = tips
        self.key_use: Counter = Counter()

        self.ingredient_recipes = [
            item for item in recipes
            if len(item.ingredients) >= 3 and item.intro
        ]
        self.step_recipes = [
            item for item in recipes
            if len(item.steps) >= 3 and item.cook
        ]
        self.overview_recipes = [
            item for item in recipes
            if item.category or item.cuisine
        ]
        self.benefit_recipes = [item for item in recipes if item.benefit_term]
        self.benefit_ingredient_recipes = [
            item for item in self.benefit_recipes if len(item.ingredients) >= 3
        ]
        self.kcal_recipes = [item for item in recipes if item.kcal]
        self.no_kcal_recipes = [item for item in recipes if not item.kcal]

        self.ingredient_cursor = Cursor([
            item["name"]
            for recipe in recipes
            for item in recipe.ingredients
            if len(item["name"]) >= 2 and item["name"] not in GENERIC_ENTITY_NAMES
        ])
        self.all_ingredients = list(dict.fromkeys(
            item["name"]
            for recipe in recipes
            for item in recipe.ingredients
            if len(item["name"]) >= 2 and item["name"] not in GENERIC_ENTITY_NAMES
        ))

        self.category_values = sorted({
            item.category for item in recipes if item.category
        })
        self.cuisine_values = sorted({
            item.cuisine for item in recipes if item.cuisine
        })
        self.tag_values = sorted({
            tag for item in recipes for tag in item.tags if tag
        })
        self.recommend_options = self._recommend_options(recipes, tips)

    # ------------------------------------------------------------------
    # 单条查询
    # ------------------------------------------------------------------

    def _recipe(self, key: str, variant: int, pool: Optional[List[Recipe]] = None) -> Recipe:
        pool = pool or self.recipes
        return pool[(variant * 37 + len(key) * 11) % len(pool)]

    def _ingredient_for(self, recipe: Recipe) -> str:
        """优先取主食材/长名字，证据里更有辨识度。"""
        candidates = [
            item["name"] for item in recipe.ingredients
            if item["name"] and item["name"] not in GENERIC_ENTITY_NAMES
        ]
        if not candidates:
            return recipe.ingredients[0]["name"]
        candidates.sort(key=lambda name: (len(name), name), reverse=True)
        return candidates[0]

    def clause(self, key: str, variant: Optional[int] = None) -> Tuple[str, List[Dict], Dict]:
        """生成一个自包含子句，返回 query / evidence / slots。"""
        if variant is None:
            variant = self.key_use[key]
        self.key_use[key] += 1

        if key == "recipe_ingredients":
            recipe = self._recipe(key, variant, self.ingredient_recipes)
            templates = [
                "{r}需要哪些食材？",
                "{r}要准备什么材料？",
                "{r}的食材清单是什么？",
                "做{r}需要买哪些东西？",
                "{r}用到的原料有哪些？",
                "我想做{r}，需要准备哪些食材？",
            ]
            query = templates[variant % len(templates)].format(r=recipe.name)
            evidence = [self._ev("recipe_ingredients", [self._ingredient_for(recipe)], [recipe.name])]
            return query, evidence, {}

        if key == "recipe_steps_time":
            recipe = self._recipe(key, variant, self.step_recipes)
            templates = [
                "{r}的步骤是什么？",
                "{r}要煮多久？",
                "{r}怎么做？",
                "{r}的烹饪时间大概多久？",
                "教我{r}的具体做法和火候",
                "{r}先做什么后做什么？",
            ]
            query = templates[variant % len(templates)].format(r=recipe.name)
            evidence = [self._ev("recipe_steps_time", [recipe.cook], [recipe.name])]
            return query, evidence, {}

        if key == "recipe_overview":
            recipe = self._recipe(key, variant, self.overview_recipes)
            templates = [
                "{r}的做法是什么？",
                "{r}是什么菜？",
                "介绍一下{r}",
                "{r}有什么特点？",
                "{r}属于什么菜系？",
                "{r}难不难做？",
            ]
            query = templates[variant % len(templates)].format(r=recipe.name)
            keyword = recipe.category or recipe.cuisine or "菜谱"
            evidence = [self._ev("recipe_overview", [keyword], [recipe.name])]
            return query, evidence, {}

        if key == "ingredient_pairing":
            pair = self.tips.pairings[variant % len(self.tips.pairings)]
            templates = [
                "{a}和{b}能一起吃吗？",
                "{a}和{b}同食会中毒吗？",
                "{a}跟{b}是不是相克？",
                "吃完{a}还能吃{b}吗？",
                "{a}和{b}搭配有什么禁忌？",
            ]
            query = templates[variant % len(templates)].format(a=pair.left, b=pair.right)
            evidence = [self._ev("ingredient_pairing", [pair.left, pair.right], [])]
            return query, evidence, {}

        if key == "ingredient_property":
            query, technique = PROPERTY_CASES[variant % len(PROPERTY_CASES)]
            evidence = [self._ev("ingredient_property", [technique], [])]
            return query, evidence, {}

        if key == "recommend_list":
            option = self.recommend_options[variant % len(self.recommend_options)]
            evidence = [self._ev(
                "recommend_list",
                option["keywords"],
                [],
                forbidden=option.get("forbidden", []),
                match=option.get("match", "any"),
            )]
            return option["query"], evidence, dict(option["slots"])

        if key == "recommend_scenario":
            people = 2 + (variant % 9)
            templates = [
                "{n}个人吃饭，推荐几个菜？",
                "我们{n}个人聚餐，怎么配菜？",
                "{n}个人吃饭，要荤素搭配",
                "家里有{n}个人，推荐一桌菜",
                "{n}个人吃，其中有个小孩，推荐几个菜",
                "{n}个人聚餐，需要加鱼吗？",
            ]
            query = templates[variant % len(templates)].format(n=people)
            evidence = [self._ev("recommend_scenario", ["配菜规则", f"人数 N={people}"], [])]
            return query, evidence, {"servings": people}

        if key == "nutrition_value":
            # 前 12 条用原文真有热量的菜；其余明确标注 no_data
            if variant < 12 and self.kcal_recipes:
                recipe = self.kcal_recipes[variant % len(self.kcal_recipes)]
                value = recipe.kcal[0][0]
                templates = [
                    "{r}热量多少？", "{r}的热量高吗？", "{r}有多少千卡？",
                    "{r}的卡路里是多少？", "{r}这道菜能量高不高？",
                ]
                query = templates[variant % len(templates)].format(r=recipe.name)
                evidence = [self._ev("nutrition_value", [value], [recipe.name])]
                return query, evidence, {}
            recipe = self._recipe(key, variant, self.no_kcal_recipes or self.recipes)
            templates = [
                "{r}热量多少？", "{r}的热量高吗？", "{r}有多少千卡？",
                "{r}的卡路里是多少？", "{r}这道菜能量高不高？",
            ]
            query = templates[variant % len(templates)].format(r=recipe.name)
            return query, [], {}

        if key == "nutrition_benefit":
            recipe = self._recipe(key, variant, self.benefit_recipes)
            templates = [
                "{r}有什么营养？",
                "{r}的营养价值怎么样？",
                "吃{r}有什么好处？",
                "{r}健康吗？",
            ]
            query = templates[variant % len(templates)].format(r=recipe.name)
            evidence = [self._ev("nutrition_benefit", [recipe.benefit_term], [recipe.name])]
            return query, evidence, {}

        if key == "food_safety":
            query, section = SAFETY_CASES[variant % len(SAFETY_CASES)]
            evidence = [self._ev("food_safety", ["食品安全", section], [])]
            return query, evidence, {}

        if key == "chitchat":
            return CHITCHAT_CASES[variant % len(CHITCHAT_CASES)], [], {}

        if key == "out_of_scope":
            return OOS_CASES[variant % len(OOS_CASES)], [], {}

        raise ValueError(f"未知路由键: {key}")

    @staticmethod
    def _ev(
        route_key: str,
        keywords: List[str],
        recipe_names: List[str],
        forbidden: Optional[List[str]] = None,
        match: str = "all",
    ) -> Dict[str, Any]:
        evidence = {
            "route_key": route_key,
            "keywords": [item for item in keywords if item],
            "recipe_names": [item for item in recipe_names if item],
            "match": match,
        }
        if forbidden:
            evidence["forbidden"] = forbidden
        return evidence

    @staticmethod
    def _clean_values(values) -> List[str]:
        """去掉“素菜,减肥餐”这类拼接值和“调料”，它们不适合做过滤条件。"""
        return sorted({
            value for value in values
            if value and "," not in value and "、" not in value and value != "调料"
        })

    @staticmethod
    def _recommend_options(recipes: List[Recipe], tips: TipsKnowledgeBase) -> List[Dict[str, Any]]:
        """推荐类查询的菜类/菜系/标签/人群选项。"""
        categories = GoldBuilder._clean_values(item.category for item in recipes)
        cuisines = GoldBuilder._clean_values(item.cuisine for item in recipes)

        options: List[Dict[str, Any]] = []
        for category in categories:
            label = "海鲜" if category == "水产" else category
            options.append({
                "query": f"推荐几道{label}",
                "slots": {"category": category},
                "keywords": [category],
                "match": "any",
            })
        for cuisine in cuisines:
            if len([item for item in recipes if item.cuisine == cuisine]) < 3:
                continue
            options.append({
                "query": f"有没有{cuisine}推荐？" if cuisine.endswith("菜") else f"有没有{cuisine}的菜推荐？",
                "slots": {"cuisine": cuisine},
                "keywords": [cuisine],
                "match": "any",
            })
        options.extend([
            {
                "query": "有什么快手菜？",
                "slots": {"time": "快手"},
                "keywords": ["快手", "简单", "10分钟"],
                "match": "any",
            },
            {
                "query": "推荐几道适合新手的菜",
                "slots": {"persona": "新手"},
                "keywords": ["新手友好", "简单"],
                "match": "any",
            },
            {
                "query": "推荐几道低热量的菜",
                "slots": {"diet": "低热量"},
                "keywords": ["低热量", "减脂", "减肥"],
                "match": "any",
            },
            {
                "query": "有什么不辣的菜推荐？",
                "slots": {"spicy": "不辣"},
                # 推荐类没有具体过滤词时，任意一份候选菜谱文档都算证据
                "keywords": [],
                "forbidden": ["辣"],
                "match": "any",
            },
        ])
        return options

    # ------------------------------------------------------------------
    # 组合
    # ------------------------------------------------------------------

    def multi_intent(self, key_a: str, key_b: str) -> Tuple[str, List[Dict], List[str], Dict]:
        query_a, ev_a, slots_a = self.clause(key_a)
        query_b, ev_b, slots_b = self.clause(key_b)
        query = f"{query_a.rstrip('？?')}？另外，{query_b}"
        slots = dict(slots_a)
        slots.update(slots_b)
        return query, ev_a + ev_b, [key_a, key_b], slots

    def same_key_two_recipes(self) -> Tuple[str, List[Dict], List[str], Dict]:
        """两道菜问同一个属性，验证多菜名展开。"""
        first = self.ingredient_recipes[self.key_use["recipe_ingredients"] % len(self.ingredient_recipes)]
        second = self.ingredient_recipes[
            (self.key_use["recipe_ingredients"] + 17) % len(self.ingredient_recipes)
        ]
        if second.name == first.name:
            second = self.ingredient_recipes[
                (self.key_use["recipe_ingredients"] + 1) % len(self.ingredient_recipes)
            ]
        self.key_use["recipe_ingredients"] += 2
        query = (
            f"{first.name}需要哪些食材？另外{second.name}需要哪些食材？"
        )
        evidence = [
            self._ev("recipe_ingredients", [self._ingredient_for(first)], [first.name]),
            self._ev("recipe_ingredients", [self._ingredient_for(second)], [second.name]),
        ]
        return query, evidence, ["recipe_ingredients", "recipe_ingredients"], {}

    def boundary_multi(self, no_retrieval_key: str, variant: int = 0) -> Tuple[str, List[Dict], List[str], Dict]:
        """域外/闲聊 + 一个正常子句，验证多意图边界。"""
        normal_key = self._pick_key(exclude=(no_retrieval_key,), pool=IN_DOMAIN_KEYS)
        normal_query, normal_ev, slots = self.clause(normal_key)
        if no_retrieval_key == "chitchat":
            # 用情绪表达而不是“你好”，避免寒暄被当成可忽略的开场白
            chit_query = BOUNDARY_CHITCHAT[variant % len(BOUNDARY_CHITCHAT)]
            query = f"{chit_query}。顺便问一下，{normal_query}"
        else:
            chit_query, _, _ = self.clause(no_retrieval_key)
            query = f"{chit_query.rstrip('？?。')}。另外，{normal_query}"
        return query, normal_ev, [no_retrieval_key, normal_key], slots

    # ------------------------------------------------------------------
    # 计数 / 配额
    # ------------------------------------------------------------------

    def _pick_key(self, exclude=(), pool: Optional[List[str]] = None) -> str:
        """取当前使用次数最少、且未到上限的键。"""
        pool = pool or ALL_KEYS
        candidates = [
            key for key in pool
            if key not in exclude and self.key_use[key] < self._cap(key)
        ]
        if not candidates:
            candidates = [key for key in pool if key not in exclude] or list(pool)
        return min(candidates, key=lambda key: (self.key_use[key], ALL_KEYS.index(key)))

    @staticmethod
    def _cap(key: str) -> int:
        # 原文热量为正的菜只有 3 道，避免把 75 条都压到 nutrition_value
        if key == "nutrition_value":
            return 45
        return 100

    def turn(
        self,
        query: str,
        route_keys: List[str],
        evidence: List[Dict],
        *,
        needs_retrieval: bool,
        response_type: str,
        slots: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        # 单意图热量问题没有原文数值时，正确行为是 no_data；组合 case 也统一走这条
        if response_type == "answer" and route_keys == ["nutrition_value"] and not evidence:
            response_type = "no_data"
        expected: Dict[str, Any] = {
            "route_keys": list(route_keys),
            "needs_retrieval": needs_retrieval,
            "response_type": response_type,
            "evidence": evidence,
        }
        if slots:
            expected["slots"] = slots
        return {"query": query, "expected": expected}

    def case(
        self,
        case_id: str,
        category: str,
        turns: List[Dict],
        *,
        boundary: bool = False,
        multi_intent: bool = False,
        multi_turn: bool = False,
        note: str = "",
    ) -> Dict[str, Any]:
        item = {
            "id": case_id,
            "category": category,
            "boundary": boundary,
            "multi_intent": multi_intent,
            "multi_turn": multi_turn,
            "turns": turns,
        }
        if note:
            item["note"] = note
        return item


def build_cases(builder: GoldBuilder) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    # 1) 300 条单轮单意图：210 条 10 个域内键 × 21，30 闲聊，30 域外，30 澄清
    for key in IN_DOMAIN_KEYS:
        for index in range(21):
            query, evidence, slots = builder.clause(key, index)
            response = "no_data" if key == "nutrition_value" and not evidence else "answer"
            cases.append(builder.case(
                f"base-{key}-{index + 1:03d}",
                "single_turn_single_intent",
                [builder.turn(
                    query, [key], evidence,
                    needs_retrieval=True, response_type=response, slots=slots,
                )],
            ))

    for index, query in enumerate(CHITCHAT_CASES):
        cases.append(builder.case(
            f"base-chitchat-{index + 1:03d}",
            "single_turn_single_intent",
            [builder.turn(
                query, ["chitchat"], [],
                needs_retrieval=False, response_type="answer",
            )],
        ))

    for index, query in enumerate(OOS_CASES):
        cases.append(builder.case(
            f"base-oos-{index + 1:03d}",
            "single_turn_single_intent",
            [builder.turn(
                query, ["out_of_scope"], [],
                needs_retrieval=False, response_type="refuse",
            )],
            boundary=True,
        ))

    for index, query in enumerate(CLARIFY_CASES):
        cases.append(builder.case(
            f"base-clarify-{index + 1:03d}",
            "single_turn_single_intent",
            [builder.turn(
                query, [], [],
                needs_retrieval=False, response_type="clarify",
            )],
            boundary=True,
        ))

    # 2) 固定多轮链路：12 条食材→步骤→热量，12 条约束→推荐
    for index in range(12):
        recipe = builder.ingredient_recipes[index % len(builder.ingredient_recipes)]
        ingredient = builder._ingredient_for(recipe)
        q1 = f"{recipe.name}需要哪些食材？"
        q2 = "那它的步骤是什么？"
        q3 = "它的热量是多少？"
        positive = bool(recipe.kcal)
        turns = [
            builder.turn(
                q1, ["recipe_ingredients"],
                [builder._ev("recipe_ingredients", [ingredient], [recipe.name])],
                needs_retrieval=True, response_type="answer",
            ),
            builder.turn(
                q2, ["recipe_steps_time"],
                [builder._ev("recipe_steps_time", [recipe.cook], [recipe.name])],
                needs_retrieval=True, response_type="answer",
            ),
            builder.turn(
                q3, ["nutrition_value"],
                [builder._ev("nutrition_value", [recipe.kcal[0][0]], [recipe.name])] if positive else [],
                needs_retrieval=True, response_type="answer" if positive else "no_data",
            ),
        ]
        cases.append(builder.case(
            f"mt-context-recipe-{index + 1:03d}",
            "multi_turn_single_intent", turns, multi_turn=True,
        ))

    constraints = ["我不吃辣", "我不能吃辣", "我忌辣", "家里人不吃辣", "我不吃辣，谢谢"]
    suggestions = [
        "推荐几道菜", "推荐几个家常菜", "今晚吃什么？推荐一下",
        "有什么菜推荐？", "给我推荐几道简单的菜",
    ]
    for index in range(12):
        constraint = constraints[index % len(constraints)]
        query = suggestions[index % len(suggestions)]
        if index >= len(constraints) * len(suggestions):
            query = f"{query}，谢谢"
        cases.append(builder.case(
            f"mt-context-constraint-{index + 1:03d}",
            "multi_turn_single_intent",
            [
                builder.turn(
                    constraint, ["chitchat"], [],
                    needs_retrieval=False, response_type="answer",
                ),
                builder.turn(
                    query, ["recommend_list"],
                    [builder._ev("recommend_list", [], [], forbidden=["辣"])],
                    needs_retrieval=True, response_type="answer",
                    slots={"spicy": "不辣"},
                ),
            ],
            multi_turn=True,
        ))

    # 3) 固定多轮边界：菜谱→域外→指代澄清
    for index in range(15):
        recipe = builder.overview_recipes[index % len(builder.overview_recipes)]
        oos = OOS_CASES[index % len(OOS_CASES)]
        cases.append(builder.case(
            f"mt-boundary-oos-{index + 1:03d}",
            "multi_turn_single_intent",
            [
                builder.turn(
                    f"{recipe.name}是什么菜？", ["recipe_overview"],
                    [builder._ev("recipe_overview", [recipe.category or recipe.cuisine or "菜谱"], [recipe.name])],
                    needs_retrieval=True, response_type="answer",
                ),
                builder.turn(
                    oos, ["out_of_scope"], [],
                    needs_retrieval=False, response_type="refuse",
                ),
                builder.turn(
                    "那它要煮多久？", [], [],
                    needs_retrieval=False, response_type="clarify",
                ),
            ],
            boundary=True, multi_turn=True,
        ))

    # 4) 固定“多轮 + 多意图”：食材+步骤，再问营养
    for index in range(12):
        recipe = builder.benefit_ingredient_recipes[
            index % len(builder.benefit_ingredient_recipes)
        ]
        ingredient = builder._ingredient_for(recipe)
        cases.append(builder.case(
            f"both-context-multi-{index + 1:03d}",
            "multi_turn_multi_intent",
            [
                builder.turn(
                    f"{recipe.name}需要哪些食材？另外要煮多久？",
                    ["recipe_ingredients", "recipe_steps_time"],
                    [
                        builder._ev("recipe_ingredients", [ingredient], [recipe.name]),
                        builder._ev("recipe_steps_time", [recipe.cook], [recipe.name]),
                    ],
                    needs_retrieval=True, response_type="answer",
                ),
                builder.turn(
                    "它有什么营养？", ["nutrition_benefit"],
                    [builder._ev("nutrition_benefit", [recipe.benefit_term], [recipe.name])],
                    needs_retrieval=True, response_type="answer",
                ),
            ],
            multi_intent=True, multi_turn=True,
        ))

    # 5) 固定“多轮 + 多意图”边界：域外+概览，相克+技法
    for index in range(10):
        recipe = builder.overview_recipes[(index * 3 + 1) % len(builder.overview_recipes)]
        oos = OOS_CASES[(index * 5 + 2) % len(OOS_CASES)]
        pair = builder.tips.pairings[index % len(builder.tips.pairings)]
        # 技法问题也用人工筛过的组合，避免“酵母怎么焯水”这种伪正例
        property_query, property_technique = PROPERTY_CASES[
            (index * 3 + 1) % len(PROPERTY_CASES)
        ]
        second_query = (
            f"{pair.left}和{pair.right}能一起吃吗？另外{property_query}"
        )
        cases.append(builder.case(
            f"both-boundary-{index + 1:03d}",
            "multi_turn_multi_intent",
            [
                builder.turn(
                    f"{oos}另外，{recipe.name}是什么菜？",
                    ["out_of_scope", "recipe_overview"],
                    [builder._ev(
                        "recipe_overview",
                        [recipe.category or recipe.cuisine or "菜谱"],
                        [recipe.name],
                    )],
                    needs_retrieval=True, response_type="answer",
                ),
                builder.turn(
                    second_query,
                    ["ingredient_pairing", "ingredient_property"],
                    [
                        builder._ev("ingredient_pairing", [pair.left, pair.right], []),
                        builder._ev("ingredient_property", [property_technique], []),
                    ],
                    needs_retrieval=True, response_type="answer",
                ),
            ],
            boundary=True, multi_intent=True, multi_turn=True,
        ))

    # 6) 85 条单轮多意图：两条正常子句
    for index in range(85):
        if index < 10:
            query, evidence, keys, slots = builder.same_key_two_recipes()
        else:
            key_a = builder._pick_key(pool=IN_DOMAIN_KEYS)
            key_b = builder._pick_key(exclude=(key_a,), pool=IN_DOMAIN_KEYS)
            query, evidence, keys, slots = builder.multi_intent(key_a, key_b)
        cases.append(builder.case(
            f"mi-normal-{index + 1:03d}",
            "single_turn_multi_intent",
            [builder.turn(
                query, keys, evidence,
                needs_retrieval=True, response_type="answer", slots=slots,
            )],
            multi_intent=True,
        ))

    # 7) 35 条单轮多意图边界：域外/闲聊 + 正常子句
    for index in range(35):
        no_retrieval_key = "chitchat" if index % 2 == 0 else "out_of_scope"
        query, evidence, keys, slots = builder.boundary_multi(no_retrieval_key, variant=index)
        cases.append(builder.case(
            f"mi-boundary-{index + 1:03d}",
            "single_turn_multi_intent",
            [builder.turn(
                query, keys, evidence,
                needs_retrieval=True, response_type="answer", slots=slots,
            )],
            boundary=True, multi_intent=True,
        ))

    # 8) 81 条普通多轮：两轮各自独立，主要看话题切换
    for index in range(81):
        key_a = builder._pick_key(pool=IN_DOMAIN_KEYS)
        key_b = builder._pick_key(exclude=(key_a,), pool=IN_DOMAIN_KEYS)
        query_a, ev_a, slots_a = builder.clause(key_a)
        query_b, ev_b, slots_b = builder.clause(key_b)
        cases.append(builder.case(
            f"mt-normal-{index + 1:03d}",
            "multi_turn_single_intent",
            [
                builder.turn(
                    query_a, [key_a], ev_a,
                    needs_retrieval=True, response_type="answer",
                    slots=slots_a,
                ),
                builder.turn(
                    query_b, [key_b], ev_b,
                    needs_retrieval=True, response_type="answer",
                    slots=slots_b,
                ),
            ],
            multi_turn=True,
        ))

    # 9) 38 条普通“多轮 + 多意图”：第一轮两个意图，第二轮一个意图
    for index in range(38):
        key_a = builder._pick_key(pool=IN_DOMAIN_KEYS)
        key_b = builder._pick_key(exclude=(key_a,), pool=IN_DOMAIN_KEYS)
        key_c = builder._pick_key(exclude=(key_a, key_b), pool=IN_DOMAIN_KEYS)
        query_a, ev_a, slots_a = builder.clause(key_a)
        query_b, ev_b, slots_b = builder.clause(key_b)
        query_c, ev_c, slots_c = builder.clause(key_c)
        cases.append(builder.case(
            f"both-normal-{index + 1:03d}",
            "multi_turn_multi_intent",
            [
                builder.turn(
                    f"{query_a.rstrip('？?')}？另外，{query_b}",
                    [key_a, key_b], ev_a + ev_b,
                    needs_retrieval=True, response_type="answer",
                    slots={**slots_a, **slots_b},
                ),
                builder.turn(
                    query_c, [key_c], ev_c,
                    needs_retrieval=True, response_type="answer",
                    slots=slots_c,
                ),
            ],
            multi_intent=True, multi_turn=True,
        ))

    return cases


def build_entities(builder: GoldBuilder) -> Dict[str, Any]:
    """给离线校验器用的实体快照。"""
    recipes = []
    for recipe in builder.recipes:
        recipes.append({
            "name": recipe.name,
            "category": recipe.category,
            "cuisine": recipe.cuisine,
            "tags": recipe.tags,
            "ingredients": [item["name"] for item in recipe.ingredients],
            "step_count": len(recipe.steps),
            "cook": recipe.cook,
            "kcal": [value for value, _ in recipe.kcal],
            "benefit_term": recipe.benefit_term,
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recipes": recipes,
        "pairings": [
            {"left": entry.left, "right": entry.right}
            for entry in builder.tips.pairings
        ],
        "safety_sections": sorted(builder.tips.safety),
        "techniques": sorted(set(builder.tips.TECHNIQUE_FILES)),
        "categories": builder.category_values,
        "cuisines": builder.cuisine_values,
        "tags": builder.tag_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "tests/gold/golden_set.jsonl"))
    parser.add_argument("--entities", default=str(ROOT / "tests/gold/entities.json"))
    args = parser.parse_args()

    driver = GraphDatabase.driver(
        DEFAULT_CONFIG.neo4j_uri,
        auth=(DEFAULT_CONFIG.neo4j_user, DEFAULT_CONFIG.neo4j_password),
    )
    try:
        recipes = load_recipes(driver)
    finally:
        driver.close()

    tips = TipsKnowledgeBase(ROOT / "data")
    builder = GoldBuilder(recipes, tips)
    cases = build_cases(builder)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    entities_path = Path(args.entities)
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    entities_path.write_text(
        json.dumps(build_entities(builder), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    key_counts = Counter()
    categories = Counter()
    for case in cases:
        categories[case["category"]] += 1
        for turn in case["turns"]:
            key_counts.update(turn["expected"]["route_keys"])

    print(f"已生成 {len(cases)} 条 -> {out_path}")
    print(f"实体快照 -> {entities_path}")
    print("分类：", dict(categories))
    print("路由键出现次数：")
    for key in ALL_KEYS:
        print(f"  {key:20s} {key_counts[key]}")
    print(
        "比例：多意图 %.1f%%，多轮 %.1f%%，边界 %.1f%%" % (
            100 * sum(1 for item in cases if item["multi_intent"]) / len(cases),
            100 * sum(1 for item in cases if item["multi_turn"]) / len(cases),
            100 * sum(1 for item in cases if item["boundary"]) / len(cases),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
