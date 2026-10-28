"""
路由相关的业务 Prompt。

集中放在这里的目的：业务 Prompt 与架构代码分离；
换领域时先改 Prompt 和 domain.py（本地规则与校验分发仍需少量同步）。
"""

from typing import Dict

from .domain import DomainConfig


UNDERSTANDING_SCHEMA = """
{
  "standalone_query": "可独立检索的问题",
  "needs_retrieval": true,
  "intents": [
    {
      "intent": "recipe_detail",
      "route_key": "recipe_ingredients",
      "confidence": 0.95,
      "span": "该意图对应的原文片段",
      "entities": [{"name": "水煮鱼", "type": "recipe"}],
      "slots": {}
    }
  ],
  "context": {
    "used_history": false,
    "topic_shift": false,
    "active_recipe": "",
    "active_topic": ""
  },
  "ambiguity": {"is_ambiguous": false, "clarify_question": ""},
  "safety": {"disclaimer_required": false}
}
""".strip()

# 按路由键追加的回答约束；generation_integration 直接取用。
ANSWER_RULES: Dict[str, str] = {
    "recipe_ingredients": "逐项列出全部食材和用量；原文标注“可选/进阶”的项要注明；禁止跨菜谱补料。",
    "recipe_steps_time": "按步骤顺序回答；时间/火候只使用原文出现的数字，没有就写“原文未标注”。",
    "nutrition_value": "只引用原文出现的数值，禁止换算或估算。",
    "ingredient_pairing": "说明相克原理、可能影响、健康建议；不做医疗诊断，并提示适量与个体差异。",
    "food_safety": "按人群/类别列出；提示特殊人群咨询医生或营养师。",
    "recommend_scenario": "先给出人数对应的菜数/荤素数量，再给候选菜；规则是“超过 8 人”才建议加鱼。",
}


def build_understanding_prompt(query: str, history_text: str, domain: DomainConfig) -> str:
    """构建“一次调用完成改写 + 多意图理解”的 Prompt。"""
    history_block = history_text.strip() if history_text else "（无历史，单轮）"

    return f"""
你是「CookWise 烹饪知识助手」的查询理解器。
你的任务只有：把用户问题理解成结构化 JSON。
不要回答问题，不要选择 hybrid / graph 这类检索引擎，不要输出解释。

用户当前问题：
{query}

最近用户轮次（只用于指代消解，可能为空）：
{history_block}

可用业务数据：
{domain.neo4j_schema}
{domain.milvus_schema}
{domain.tips_sources}

顶层意图（只能从这 6 个里选）：
{', '.join(domain.top_intents)}

路由键（只能从下面这些里选；一个意图对应一个路由键）：
{domain.route_table_text()}

判断规则（按顺序执行）：
1. 先判断 needs_retrieval。日常闲聊、情绪表达、寒暄、非饮食领域 -> false；菜谱/食材/推荐/营养/食品安全 -> true。
   即使闲聊里出现真实菜名（如“宫保鸡丁和红烧肉哪个好吃”），needs_retrieval 也必须是 false。
   与饮食完全无关的问题 route_key 要用 out_of_scope，不要用 recipe_overview 兜底。
   讲故事、讲笑话、唱歌等娱乐请求 -> out_of_scope。
2. 相克/同食/能不能一起吃 -> ingredient_pairing，证据源是 tips 文档，不要走 Neo4j 关系。
3. 精确数值（千卡/几分钟/多少克）-> nutrition_value 或 recipe_steps_time；原文没有就应回答“无数据”，不要估算。
   问营养/功效/好处 -> nutrition_benefit；即使菜名不常见，也不要判成域外（如“海边落日有什么营养？”仍算 nutrition_benefit）。
4. 只问“X 的做法是什么 / 介绍一下 X / X 是什么菜 / X 有什么特点”-> recipe_overview；
   明确说“步骤 / 怎么做 / 怎么煮 / 怎么炒 / 多久 / 火候”才 -> recipe_steps_time。
   问“介绍/特点”时，即使菜名里带“煮/蒸/锅/酥”等字（如“介绍一下油酥”“煮锅蒸米饭有什么特点”），也仍 -> recipe_overview。
   食材处理技法（去腥/焯水/腌制/凉拌/微波炉/空气炸锅/高压锅/挑选/储存/怎么切）
   -> ingredient_property；问的是食材怎么处理，不要归 recipe_steps_time 或 food_safety。
5. 求一组菜/带约束 -> recommend_list；问人数、荤素、加鱼 -> recommend_scenario；问单道菜 -> recipe_*。
   “10个人吃饭，要荤素搭配”只算 recommend_scenario，不要再拆 recommend_list。
   按菜类筛选时把“素菜/荤菜/汤类/主食/甜品/水产/早餐/凉菜/饮料”放进 slots.category；“减脂/低热量”放进 slots.diet，两者不要混用。
6. 情绪表达/自我陈述（“好想吃甜的”“我刚开始学做菜”）-> chitchat；
   明确请求（“推荐几道甜的”“推荐几道新手菜”）-> recommend_list。
7. 多意图：intents 可以是多项；每项必须有 span。只有证据源不同的部分才拆，不要无条件拆。
   情绪表达/闲聊 + 菜谱任务（如“今天好累，顺便问一下宫保鸡丁怎么做”）要拆成 chitchat + 任务两项；
   纯寒暄（如“你好”）可以忽略，不单独算一项。
8. 实体只填用户明确提到的；不要编造图谱里没有的菜名/食材。
9. 指代消解：把“它/这个/那要多少盐”改写成 standalone_query，但不要改变原意。
10. 多个候选实体或指代无法解析时，is_ambiguous=true，并给一个简短的 clarify_question。
11. 食品安全（没熟/发芽/发霉/变质/寄生虫/过敏/孕妇或小孩能不能吃/中心温度/怎么算熟透）
    -> food_safety，不要走 nutrition_value；相克、食品安全、健康建议类问题，safety.disclaimer_required=true。

输出要求：
- 只输出一个 JSON 对象，不要 Markdown 代码围栏，不要前后解释。
- 字段必须完整；没有内容用空字符串、空对象或空数组。
- confidence 是 0~1 的小数。
- 证据形态、完整性和来源由系统按 route_key 自动补齐，不需要你输出。

JSON 结构：
{UNDERSTANDING_SCHEMA}
""".strip()


def build_repair_prompt(raw_output: str, error: str, domain: DomainConfig) -> str:
    """结构化输出修复：把错误和原始输出一起回灌，只重试一次。"""
    return f"""
你上一次输出的 JSON 无法通过校验：{error}

请只输出修正后的 JSON，不要解释、不要代码围栏。
可用路由键：
{domain.route_table_text()}

必须符合这个结构：
{UNDERSTANDING_SCHEMA}

上一次的原始输出：
{raw_output}
""".strip()


def build_direct_answer_prompt(question: str, route_key: str) -> str:
    """不需要检索时的直接回答 Prompt（闲聊 / 域外）。"""
    if route_key == "out_of_scope":
        return f"""
你是 CookWise 烹饪助手。用户的问题超出了菜谱、食材、营养、厨房技巧的范围。
请礼貌说明你只能回答烹饪相关问题，并用一句话引导用户回到这些话题。
不要回答原问题，不要编造事实。
用户问题：{question}
""".strip()

    return f"""
你是 CookWise 烹饪助手。用户正在进行日常闲聊，不需要检索知识库。
请用自然、简短、友好的中文回应；不要编造具体菜谱、用量或营养数字。
可以轻轻把话题引回做饭，但不要生硬。
用户：{question}
""".strip()


def build_no_evidence_prompt(question: str) -> str:
    """检索没有拿到证据时的兜底 Prompt。"""
    return f"""
你是 CookWise 烹饪助手。检索没有找到能回答用户问题的证据。
请明确告诉用户“知识库暂未收录这个信息”，不要猜测、不要编造菜谱或数值。
可以建议用户换一种问法，或问一道具体的菜。
用户问题：{question}
""".strip()
