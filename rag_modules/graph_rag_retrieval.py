"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配。

预览：
这个文件定义了一个类 `GraphRAGRetrieval`，它是图RAG检索的核心。对外它主要提供三个「入口」方法：
1. `initialize()`：连接 Neo4j 并预热图索引（后续所有图操作都靠它建立的 `self.driver`）。
2. `graph_rag_search(query, top_k)`：主搜索接口。内部完整走一遍「理解意图 → 按类型分发 → 图结构排序」的流程，最后返回一批 `Document` 对象。
3. `understand_graph_query(query)`：单独把「意图理解」这一步暴露出来，方便测试和路由器复用。
内部还藏着一整套「发动机」：`multi_hop_traversal`（多跳遍历）、`extract_knowledge_subgraph`（子图提取）、
`graph_structure_reasoning`（图推理）、`adaptive_query_planning`（自适应规划），以及一票下划线开头的辅助方法。

===============================================================================
实现说明（写给维护者）
===============================================================================
本模块面向的真实图谱是「中式烹饪知识图谱」，其结构经过实测确认，若干特性直接决定了这里的算法取舍，
把这些前提写下来，是为了避免后续维护者按「常识」想当然地改坏它：

1. 【食材是按菜谱私有的】Ingredient 节点 2906 个，而 REQUIRES 边仅 2905 条，且每个食材节点度数几乎恒为 3
   （REQUIRES + BELONGS_TO_CATEGORY + HAS_CONCEPT_TYPE）。也就是说**不存在跨菜谱共享的食材节点**，
   「食材 → 菜品 → 同类食材」这种常见假设在本图谱上【走不通】。
   菜谱之间真正可用的连通通路只有：
       Recipe -[:BELONGS_TO_CATEGORY]-> Category <-[:BELONGS_TO_CATEGORY]- Recipe   （同分类，最主要）
       Recipe -[:SIMILAR]- Recipe                                                     （稀疏，仅约 46 道菜有）
       Recipe -[:CONTAINS_STEP]-> CookingStep -[:USES_SAME_METHOD]-(...)- Recipe      （同技法，精细）
       以及 cuisineType / difficulty 这类属性级同组关系
2. 【hub 节点必须剪枝】ConceptType 'Ingredient' 度数 2906、Category '调料' 1556、CookingStep '步骤2' 511。
   不做度数剪枝的话，度数 3 的变长路径实测会产生 127214 条路径。
3. 【变长路径与 shortestPath 的上界不能参数化】Neo4j 明确报错：「Parameter maps cannot be used in MATCH patterns」。
   因此深度只能以【经过钳制的整数】内插进 Cypher 字符串（见 `_MAX_TRAVERSAL_DEPTH` 与 `_clamp_int`），
   这正是本模块所有深度参数都强制走 `_clamp_int` 的原因。
4. 【关系类型在 Python 侧只能用 rel.type 取】Cypher 里的 `type(rel)` 才是类型；而 neo4j 驱动的 Relationship 对象
   上是 `rel.type` 属性，`type(rel).__name__` 只会得到字符串 'Relationship'（这是本模块修掉的一个真实缺陷）。
5. 【脏数据】图谱里混有 GraphStatistics / RelationshipStatistics / Root / CookingMethod / CookingTool 等统计与层次节点，
   以及 nodeId 为 '200000000'（name='菜谱'）、'300000000'（name='食材'）的汇总占位节点。
   所有查询都通过标签白名单 + nodeId 黑名单过滤（见 `_MEANINGFUL_LABELS` / `_PLACEHOLDER_NODE_IDS`）。
6. 【LLM 是推理模型】本项目的 deepseek-flash 会先输出 reasoning_content，若不关闭思考，max_tokens 会被推理过程吃光，
   返回的 content 为空字符串。因此所有 LLM 调用都必须带 `extra_body={"thinking": {"type": "disabled"}}`，
   并且对返回内容做「剥代码围栏 + 容错 JSON 解析 + 空内容判定」（见 `_call_llm_json` / `_extract_json_object`）。
"""


import json
import logging
import re
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# =============================================================================
# 图结构常量：集中定义，便于按数据变化调整
# =============================================================================

# 具有业务语义的节点标签白名单。
# 只有这些标签的节点才允许进入检索结果；用于滤掉 GraphStatistics / Root / CookingMethod 等统计与层次节点。
_MEANINGFUL_LABELS: Tuple[str, ...] = (
    "Recipe",
    "Ingredient",
    "CookingStep",
    "Category",
    "ConceptType",
    "DifficultyLevel",
)

# 【可作为路径中转的标签】——刻意不含 ConceptType 与 DifficultyLevel。
# 原因：这两类是纯分类学节点，作为「中转站」时只会产生毫无业务意义的捷径：
#   ConceptType 'Ingredient' 度数 2906，它会把任意两个节点连成 2 跳「路径」；
#   DifficultyLevel '四星' 度数 78，于是「宫保鸡丁 → 四星 → 水煮鱼」会被当成一条路径证据返回，
#   而它的实际含义只是「两道菜都是四星」。
# 它们仍保留在 _MEANINGFUL_LABELS 中（可以作为直接邻居/端点被展示，例如「这道菜是四星」），
# 但不再参与路径匹配。注意这两个标签的 nodeId 也是 NULL（ConceptType 全为 NULL），
# 因此也无法通过 _attach_degrees 拿到度数、无法被 _is_hub 剪枝——从白名单剔除是唯一可靠的拦截点。
_TRAVERSAL_LABELS: Tuple[str, ...] = ("Recipe", "Ingredient", "CookingStep", "Category")

# 路径类查询的标签白名单片段（用于 nodes(path) 过滤）
_PATH_LABEL_FILTER_X: str = " OR ".join(f"x:{label}" for label in _TRAVERSAL_LABELS)

# 汇总占位脏节点的 nodeId（见 data/cypher/nodes.csv 的 200000000 / 300000000），必须排除
_PLACEHOLDER_NODE_IDS: Tuple[str, ...] = ("200000000", "300000000")

# 【NULL 安全的占位节点过滤片段】——这是本模块踩过的一个重大陷阱，务必不要改回简写形式。
# 图谱实测：19 个 Category 与 8 个 ConceptType 节点的 nodeId 全部为 NULL，
# 而 Cypher 的三值逻辑里 `NULL IN [...]` 求值为 NULL、`NOT NULL` 仍是 NULL，WHERE 把 NULL 当 false
# → 整行被丢弃。于是简写 `NOT x.nodeId IN $placeholder_ids` 会【静默地把所有经过 Category 的路径全部删除】：
# 实测宫保鸡丁 2 跳路径 600 → 126 条、其中含 Category 的路径 119 → 0 条；水煮鱼含 Category 路径 19119 → 0 条。
# 而「同 Category 的兄弟菜谱」正是本图谱【最主要】的跨菜谱通路（食材是按菜谱私有的，走不通），
# 所以这个简写会把模块的核心能力整体废掉，且不报错、不打日志。
# 用 coalesce 把 NULL 归一成空串后再比较：真实的占位节点（nodeId 200000000/300000000）照旧被排除，
# nodeId 为 NULL 的业务节点（Category/ConceptType）则不再被误杀。
# 具体写法见各条 Cypher 中的 `NOT coalesce(x.nodeId, '') IN $placeholder_ids`（共 10 处）。

# 占位节点的名称，用于内存侧兜底过滤
_PLACEHOLDER_NAMES: Set[str] = {"菜谱", "食材"}

# 关系类型语义权重：数值越高代表该关系对「烹饪知识推理」的贡献越大。
# 权重来自实测的图谱结构与业务语义，而非拍脑袋：
#   REQUIRES / CONTAINS_STEP 是菜谱的本体组成，信息量最高；
#   USES_SAME_METHOD 是跨菜谱的精细通路（实测同技法邻居仅 2 个，精确度高）；
#   DIFFICULTY_LEVEL 同层多达 76 道菜，区分度低；HAS_CONCEPT_TYPE 几乎全库互联（322 道菜同属 'Recipe'），几乎无区分度。
_RELATION_WEIGHTS: Dict[str, float] = {
    "REQUIRES": 1.0,
    "CONTAINS_STEP": 0.9,
    "NEXT_STEP": 0.85,
    "SIMILAR": 0.8,
    "BELONGS_TO_CATEGORY": 0.7,
    "USES_SAME_METHOD": 0.65,
    "USES_SAME_TOOL": 0.5,
    "DIFFICULTY_LEVEL": 0.35,
    "HAS_CONCEPT_TYPE": 0.2,
}

# 未知关系类型的默认权重（既不该被当作噪声丢弃，也不该被高估）
_DEFAULT_RELATION_WEIGHT: float = 0.3

# hub 剪枝阈值：度数超过它的【中间节点】视为 hub，路径将不再穿越它。
# 实测：ConceptType 'Ingredient'=2906、Category '调料'=1556、CookingStep '步骤2'=511，都远超该阈值。
_HUB_DEGREE_THRESHOLD: int = 120

# 图遍历深度上限。Cypher 中变长路径的深度只能内插字面量，因此必须先把用户/LLM 给的深度钳制在这个范围内。
_MAX_TRAVERSAL_DEPTH: int = 3

# 【3 跳展开的种子度数门槛】只对"小种子"放开 3 跳，度数超过它的种子降到 2 跳。
#
# 为什么 _HUB_DEGREE_THRESHOLD 不够用：那个阈值是"要不要彻底禁止"，是个开关；
# 而变长路径的枚举量近似随种子度数【相乘】增长，度数 80 的种子即使没到 hub 级别，
# 3 跳展开也已经要枚举上万行。实测（多跳 + 目标关键词 '蔬菜'，同一台机器）：
#     种子度数  3  -> 枚举   149 行、  68ms  |  3 跳相比 2 跳【新增 32 个节点】
#     种子度数 17  -> 枚举  1327 行、  80ms  |  新增 0 个节点
#     种子度数 27  -> 枚举  3519 行、 984ms  |  新增 0 个节点
#     种子度数 64  -> 枚举  7256 行、 951ms  |  新增 1 个节点
#     种子度数 83  -> 枚举 17866 行、4722ms  |  新增 2 个节点
# 即：3 跳的收益集中在"小种子"上（小菜谱的第三跳才真的走到新内容），
# 而密度大的种子第三跳只是在已经覆盖过的邻域里绕圈，代价却是 50 倍以上。
# 所以 12 这个阈值不是拍脑袋，而是"收益出现 / 消失"的分界附近取的保守值：
# 度数 <= 12 的种子（含全部食材种子——食材节点按菜谱复制，度数恒为 3）保留 3 跳能力。
_DEEP_SEED_DEGREE_LIMIT: int = 12

# 单次多跳检索返回的路径条数上限，防止结果膨胀
_MAX_PATH_RESULTS: int = 60

# 参与【深遍历（>=2 跳）】的种子个数上限。
# 原因：UNWIND 多个种子时，每个种子都会各自完整枚举一遍变长路径，成本线性叠加，
# 而 LIMIT 只截断返回、不截断枚举。实测 1 个种子 145ms vs 5 个种子 4394ms，返回的都是那 60 条。
_MAX_DEEP_SEED_COUNT: int = 2

# 子图节点预算的合理区间
_MIN_SUBGRAPH_NODES: int = 10
_MAX_SUBGRAPH_NODES: int = 200

# `_build_graph_index` 预热缓存的节点规模上限
# 【为什么是 20000 而不是 5000】实测符合白名单的业务节点共 5746 个，而 5000 这个上限会
# 静默截断掉 746 个节点（被截掉的全部是度数 3 的叶子：115 个 Ingredient + 1 个 CookingStep，
# 对应 108 个不同名字，例如「泡发好的海参（北极参）」「大葱葱白」）。由于 ORDER BY 只按
# degree DESC、并列值之间没有稳定次序，"保住哪 5000 个"由存储层决定，同一份数据两次预热
# 可能得到不同的索引。后果不是"少几个节点"，而是这些名字在 entity_cache / name_index 里
# 彻底不存在：LLM 不可用走本地兜底抽实体时，「泡发好的海参（北极参）能做什么菜」会因为
# 抽不出实体而整条链路返回 0 条结果。上限抬到远大于实际规模（单个节点字典约 200 字节，
# 5746 个约 1MB）后截断永不触发，索引覆盖完整。
_GRAPH_INDEX_LIMIT: int = 20000

# 各类缓存容量上限，避免长时间运行后内存无界增长
_QUERY_CACHE_LIMIT: int = 256
_SUBGRAPH_CACHE_LIMIT: int = 64

# 多跳/推理意图的判别模式：命中越多，查询越可能需要深遍历
_MULTI_HOP_PATTERNS: Tuple[str, ...] = (
    r"配(什么|哪些|啥|点)",
    r"和.{1,10}(一起|搭配|能|可以|关系)",
    r"为什么",
    r"怎么(做|样|办)",
    r"哪些",
    r"什么关系",
    r"区别",
    r"类似",
    r"相似",
    r"通过",
    r"间接",
    r"影响",
    r"原因",
    r"除了",
    r"还可以",
    r"推荐",
    r"适合",
)

# 多实体连接词：出现说明查询同时在谈多个对象，复杂度上升
_CONNECTIVE_CUES: Tuple[str, ...] = ("和", "跟", "与", "还是", "以及", "都", "比较", "相比", "同时")

# 复杂度评分的四个分量权重，合计为 1.0
_COMPLEXITY_WEIGHT_LENGTH: float = 0.25
_COMPLEXITY_WEIGHT_MULTIHOP: float = 0.35
_COMPLEXITY_WEIGHT_CONNECTIVE: float = 0.30
_COMPLEXITY_WEIGHT_PUNCT: float = 0.10

# 复杂度阈值：低于 SIMPLE 走 1 跳，低于 MEDIUM 走多跳，否则子图 + 推理（与 adaptive_query_planning 的分支保持一致）
_COMPLEXITY_SIMPLE: float = 0.3
_COMPLEXITY_MEDIUM: float = 0.7

# Document 的 search_type 先验分：知识子图信息最完整，聚类结果最弱
_SEARCH_TYPE_PRIOR: Dict[str, float] = {
    "knowledge_subgraph": 0.85,
    "graph_entity_relation": 0.80,
    "graph_path": 0.75,
    "graph_cluster": 0.70,
}

# 最终融合排序的权重（词法证据 / 图结构 / 结果类型先验 / 可溯源程度），合计为 1.0
_FUSE_WEIGHT_LEXICAL: float = 0.40
_FUSE_WEIGHT_STRUCT: float = 0.25
_FUSE_WEIGHT_PRIOR: float = 0.20
_FUSE_WEIGHT_TRACE: float = 0.15

# 词法重合度里 bigram 与 unigram 的配比：bigram 更能代表短语级匹配，权重更高
_LEXICAL_BIGRAM_WEIGHT: float = 0.6

# 中文停用字符：计算词法重合度前先剔除，避免虚词把相似度稀释掉
_STOPWORD_CHARS: str = "的了和与跟及以及是有在就都也还很太更最把被让使对为从到向于吗呢吧啊呀哦嗯什么怎么如何哪些哪这个那个我你他她它们之其一二三四五六七八九十"

# 「目标实体」后缀噪声：LLM 常产出「蔬菜类食材」这类"类型描述"而非真实实体名，需要剥离后再匹配
_TARGET_SUFFIX_NOISE: Tuple[str, ...] = (
    "类食材", "类菜品", "类原料", "类型的食材", "类别的食材", "类别的菜品",
    "食材类", "菜品类", "类型", "类别", "类", "食材", "原料", "菜品",
)

# 纯「元类型词」：它们只说明"想找哪一类东西"，不含任何可区分的语义。
# 这类词绝不能拿去硬过滤邻居——实测 target=['食材'] 时，过滤条件 "邻居名/分类包含『食材』"
# 会把「宫保鸡丁需要哪些食材」的全部 31 个邻居筛成 0 条，即"问题本身被约束条件吃掉了"。
# 注意：这里的清单必须【只含元类型词】。「蔬菜」「调料」「蛋白质」是图谱里真实存在的 Category 节点，
# 具备区分度，必须保留其过滤作用。
_GENERIC_TARGET_KEYWORDS: Set[str] = {
    "食材", "原料", "材料", "菜品", "菜谱", "菜", "类型", "类别", "种类",
    "东西", "内容", "信息", "知识", "实体", "节点", "关系", "概念",
}

# -----------------------------------------------------------------------------
# 【实体词表对齐】用户用词与图谱标准词之间的系统性错位
# -----------------------------------------------------------------------------
# 实测「鸡肉配什么蔬菜好？」的召回缺口几乎全部来自词表错位，而不是图谱没有数据：
#   * 名字【正好等于】「鸡肉」的 :Ingredient 节点只有 1 个；
#   * 名字【包含子串「鸡肉」】的只有 2 个（另一个是「无骨肉（猪肉、牛肉、鸡肉、鱼丸、火腿肠）」）；
#   * 而图谱里真正属于鸡肉的可食用部位有 12 个不同名字：
#     鸡肉 / 鸡胸肉 / 鸡胸脯肉 / 鸡腿 / 鸡腿肉 / 鸡翅中 / 鸡全翅 / 鸡爪 / 无骨鸡爪 / 大鸡腿 / 生鸡肝 / 农村玉米鸡；
#   * 对齐后「鸡肉」经菜谱共现可召回 27 种蔬菜，不对齐则只有 1 种（青菜）。
#
# 对齐判定用三条【通用信号】，不引入任何菜名/菜系硬编码词表：
#   1. 头语素相同：候选名与查询词共享头 morpheme（「鸡肉」->「鸡」），保证同物种/同主料；
#   2. 同食材类目：候选名的 category 与查询词命中的参照节点一致，挡住跨料类的词
#      （「鸡精」「鸡汤」的 category 是「调料」，与「蛋白质」不同类，因此被挡掉）；
#   3. 可食用部位语素收尾：候选名末字必须落在下面的集合里。
#      这一条专门解决「鸡蛋」与「鸡肉」的歧义：二者共享头语素「鸡」且同属「蛋白质」，
#      只有"末字是不是可食用部位"能把它们分开（蛋/清/黄/精/粉/汤 都不是）。
#
# 为什么必须有第 3 条，而不能只用"衍生品后缀黑名单"：
# 实测黑名单会漏掉「摊鸡蛋皮」（末字「皮」）与「蛋类（鸡蛋鸭蛋等）」（末字「）」），
# 让两个非鸡肉项混进「鸡肉」的对齐集合；改用部位语素白名单后，
# 在 12 个真实鸡肉食材上精确率 12/12、召回 12/12。
#
# 集合同时收录【部位语素】（肉腿翅爪肝腩）与【切割形态语素】（片丝丁末块条卷），
# 因为「猪肉末」「肥牛片」「火锅牛肉卷」这类名字同样是真实食材。
# 之所以敢收得较宽：候选还必须先通过"共享头语素"这一关，
# 即「蒜末」「姜末」「土豆片」这些名字根本进不了「鸡肉」的候选集。
_EDIBLE_PART_MORPHEMES: Set[str] = set(
    "肉腿翅爪肝脯排骨脊腩肘蹄掌胗舌尾脖头中根尖块子仁柳末丝丁片段条粒卷"
    "鸡鸭鹅猪牛羊蛙鱼虾蟹贝"
)

# 【物种语素】单独一份：用于"首字兜底"这种更粗的放宽手段（见 _expand_alias_names）。
# 它与 _EDIBLE_PART_MORPHEMES 的区别是【只含物种、不含部位/切法】：
# 「虾」是物种（虾仁 -> 虾类各名字，语义不收窄），而「排」只是切割形态
# （排骨 -> 羊排/牛排 会把猪肉换成别的肉，属于语义漂移）。两份集合分开，才能对
# "末字该是部位"和"首字该是物种"这两种判定用不同宽严标准。
_SPECIES_MORPHEMES: Set[str] = set("鸡鸭鹅猪牛羊蛙鱼虾蟹贝")

# 「材料通称」语素：查询词以它收尾时，说明用户说的是"[主料]+材料通称"
# （「鸡肉」「猪肉」「牛肉」「羊肉」），其同族是"该主料的各种部位/切法"，
# 因此头语素要取【去掉这个通称之后的主料语素】（「鸡肉」->「鸡」）。
# 实测：若不移除，「猪肉」的头语素是「猪肉」，候选名必须包含连续的「猪肉」三个字，
# 于是「猪五花肉」「猪里脊」「猪蹄」全部落空（对齐结果为空）；移除后能覆盖 11 个猪肉食材名。
# 反之，「鸡蛋」「虾仁」这类末字是【具体制品/部位】的词不能移除末字，
# 否则「鸡蛋」会被扩到「鸡腿肉」上去。所以这里只登记"材料通称"，不登记具体部位。
_GENERIC_MATERIAL_MORPHEMES: Set[str] = {"肉"}

# 稀疏触发阈值：只有"精确命中 + 子串命中"的不同名字数【少于】该值时，才启用同族扩展。
# 依据（实测名字数）：鸡肉 2、羊肉 2、牛肉 7、鸡蛋 7、虾 12、鱼 19、菠菜 6、面粉 5。
# 取 8 是为了让 鸡肉/羊肉/牛肉（同族部位名丰富、子串命中稀少的动物性食材）都能被扩展，
# 同时不至于对「鱼」这类本来就有 19 个名字的词做无谓扩展。
_ALIAS_SPARSITY_THRESHOLD: int = 8

# 单个查询词最多补入的同族别名数。成本上界：别名数直接线性放大后续遍历的枚举量，
# 实测本图谱里动物性食材的同族名都在 12 个以内，故取 12。
_ALIAS_MAX_NAMES: int = 12

# 同族别名最多折算成多少个 nodeId。别名按 nodeId 定位（走 ingredient_id_unique 索引），
# 每个别名名下有 1~3 个反规范化副本节点，取 64 足以覆盖 12 个别名的全部副本，
# 同时给"某个别名名下有异常多副本"留了硬上界。
_ALIAS_MAX_NODE_IDS: int = 64

# -----------------------------------------------------------------------------
# 【搭配共现检索】"A 配什么 B"这类问句的结构化通道
# -----------------------------------------------------------------------------
# 变长路径 (source)-[*1..N]-(target) 对"搭配"类问句是钝的：它不区分"经由什么连接"，
# 只能靠事后 hub 剪枝与 LIMIT 收口。实测从 12 个鸡肉食材出发做 3 跳变长展开会枚举
# 3963 行、耗时 680ms，且返回的 60 条路径被"最短路径优先"挤在少数几个菜谱上
# （这正是「鸡肉配什么蔬菜」只出 2 条结果的原因之一）。
# 而"经由菜谱共现"是有向、有类型的两跳结构：
#     (食材A) <-[:REQUIRES]- (菜谱) -[:REQUIRES]-> (食材B)
# 它天然回答了"搭配"问句，且可在 Cypher 内按【共现菜谱数】聚合排序。
# 实测同场景（12 个鸡肉食材 -> 蔬菜）仅 6~34ms，并给出按共现强度排序的 27 种蔬菜。
#
# 触发条件：源种子含 :Ingredient、目标关键词非空、depth >= 2 —— 三条同时成立才走此通道。
# 目标关键词为空时不走：实测退化版（不过滤目标）会把「可乐」「啤酒」「凉白开水」「丽滋饼干」
# 一并当作"搭配"返回，噪声过大，不如让原通道保持不变。
_PAIRING_MIN_RESULTS: int = 3           # 产出少于此数就认为该通道不适用，完整交还原通道
_PAIRING_BUDGET_RATIO: float = 0.5      # 命中时最多占用多少比例的路径预算（其余留给原通道）
_PAIRING_ROW_FACTOR: int = 4            # 单次取回行数 = 结果上限 × 该系数（同名字节点需去重归并）
_PAIRING_WITNESS_LIMIT: int = 3         # 每个搭配最多收集几条"见证"三元组（Python 侧取确定性的一条）

# 共现强度对相关性分的加成：结构完全相同的搭配路径（食材-菜谱-食材，2 跳）用
# _score_path 打分是【完全同分】的，唯一能区分它们的信号就是"这个搭配有多少个菜谱支持"。
# 因此把共现菜谱数折成一个有界加成加到 relevance_score 上。
# 之所以要有界（封顶 0.2）：避免"高频搭配"压过结构证据本身，
# 也让排序在同分时退回稳定排序的输入顺序（= 本方法给出的 support 降序）。
_PAIRING_SUPPORT_BONUS: float = 0.2
_PAIRING_SUPPORT_SATURATION: float = 5.0


# =============================================================================
# 模块级辅助函数
# =============================================================================


def _dedup_preserve_order(items: List[Any]) -> List[Any]:
    """去重并保持首次出现的顺序（用于稳定结果顺序，避免集合无序导致输出抖动）。"""
    seen: Set[Any] = set()
    result: List[Any] = []
    for item in items:
        if item is None or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _content_chars(text: str, drop_stopwords: bool = True) -> List[str]:
    """抽取文本中的"内容字符"：去掉空白、标点与（可选的）中文虚词。"""
    if not text:
        return []
    chars = [ch for ch in str(text) if ch.strip() and not _is_punctuation(ch)]
    if drop_stopwords:
        chars = [ch for ch in chars if ch not in _STOPWORD_CHARS]
    return chars


def _is_punctuation(ch: str) -> bool:
    """判断字符是否为标点或符号（覆盖中英文标点），用于词法处理时剔除噪声。"""
    return bool(re.match(r"[\s\W_]", ch, flags=re.UNICODE)) and not re.match(r"[一-鿿]", ch)


def _char_bigrams(text: str) -> Set[str]:
    """把文本切成字符 bigram 集合。

    中文没有天然词边界，字符 bigram 是无需分词即可稳定度量「短语级重合」的折中方案：
    例如「川菜特色」-> {'川菜','菜特','特色'}，能与知识子图描述里的「川菜」直接对上。
    """
    chars = _content_chars(text)
    if len(chars) < 2:
        return set(chars)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _char_unigrams(text: str) -> Set[str]:
    """把文本切成内容字符的单字集合，作为 bigram 的补充（提升短查询的召回）。"""
    return set(_content_chars(text))


def _containment(needle: Set[str], haystack: Set[str]) -> float:
    """集合包含度：needle 中有多大比例出现在 haystack 中，取值 [0,1]。"""
    if not needle:
        return 0.0
    return len(needle & haystack) / float(len(needle))


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    """把任意输入安全地钳制成 [low, high] 内的整数。

    这是本模块的安全基石：Neo4j 不允许把变长路径深度参数化，深度必须以字面量内插进 Cypher，
    所以【任何】来自 LLM 或调用方的深度值都必须先经过它，避免出现 `[*1..10]` 这种把库拖垮的查询。

    【为什么必须捕获 OverflowError】Python 的 json.loads 接受非标准字面量，
    `json.loads('{"max_depth": 1e999}')` 与 `json.loads('{"max_depth": Infinity}')` 都会得到 float('inf')，
    而 `int(float('inf'))` 抛的是 OverflowError 而不是 ValueError。
    只捕 (TypeError, ValueError) 会让这个异常穿透 _build_graph_query_from_llm → understand_graph_query →
    一路逃出 graph_rag_search（后者的 try 从图查询阶段才开始，不覆盖意图理解那一步），
    最终把异常抛给用户，而不是按设计降级为本地启发式查询。
    """
    try:
        ivalue = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if ivalue < low:
        return low
    if ivalue > high:
        return high
    return ivalue


def _coerce_query_text(query: Any) -> str:
    """把任意输入安全地规范成「查询文本」（字符串）。

    【为什么需要】本模块多个公开入口会对 query 直接调用 `.strip()`（understand_graph_query
    的缓存键、_analyze_query_complexity 的空值判断），只兜住 None 是不够的：
    实测 graph_rag_search(123 / ["a"] / {"a": 1} / 1.5) 都会抛
    `AttributeError: 'int' object has no attribute 'strip'`，而该调用发生在
    graph_rag_search 的 try 之前，异常直接穿透公开入口抛给调用方 ——
    对"检索模块应当降级不抛错"的既有约定来说是破坏性的。

    语义：str 原样返回（不做 strip，保持调用方拿到的就是自己的输入）；
    其它类型统一转成字符串，None 转空串 —— 空串会被下游当作"没有有效查询"而安全返回空结果。
    """
    if isinstance(query, str):
        return query
    if query is None:
        return ""
    try:
        return str(query)
    except Exception:
        # str() protocol 被重写成会抛异常的极端情形：退化为空查询而不是抛错
        return ""


class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"              # 多跳查询：A通过什么连接到C？
    SUBGRAPH = "subgraph"                # 子图查询：A相关的所有信息
    PATH_FINDING = "path_finding"        # 路径查找：从A到B的最佳路径
    CLUSTERING = "clustering"            # 聚类查询：和A相似的都有什么？

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]

class GraphRAGRetrieval:
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

        # 意图理解结果缓存：key = 原始查询串，value = GraphQuery。
        # 同一个问题常被路由器与生成阶段反复问到，缓存可省掉重复的 LLM 调用。
        self._query_understanding_cache: Dict[str, "GraphQuery"] = {}

        # 实体名 -> nodeId 列表 的反向索引，由 _build_graph_index 预热，用于把自然语言里的实体名快速落到图谱节点上
        self._name_index: Dict[str, List[str]] = defaultdict(list)

        # 【属性值词表】菜系（Recipe.cuisineType）、分类（category）这类【只作为属性存在】的值。
        # 它们在图谱里没有同名节点（实测「川菜」不存在同名节点，只存在于 33 个 Recipe 的 cuisineType），
        # 因此不在 _name_index 里；但 _resolve_seed_nodes 的属性分支【能】匹配到它们。
        # 单独一张表而不是混进 _name_index：后者的契约是"名称 -> nodeId"，混入属性值会污染别名定位。
        self._vocab_phrases: Set[str] = set()

        # 子图缓存的插入顺序，用于容量超限时做 FIFO 淘汰（无需引入额外依赖）
        self._subgraph_cache_order: deque = deque()

        # 最近一次多跳扩展【实际使用】的深度（由 _expand_multi_hop_paths 回填）。
        # 请求深度与实际深度可能不同（种子度数会让深度降级），而"兜底查询是否与主路径等价"
        # 必须按实际深度判断，否则会原样重跑一遍刚跑过的查询。
        self._last_effective_depth: Optional[int] = None

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

    # -------------------------------------------------------------------------
    # 基础设施：连接判定、安全查询、节点/关系序列化
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # 图索引与实体解析
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # 意图理解
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # 多跳遍历
    # -------------------------------------------------------------------------

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳图遍历：这是图RAG的核心优势
        通过图结构发现隐含的知识关联
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")

        paths: List[GraphPath] = []

        if not self._is_driver_ready():
            logger.error("Neo4j连接未建立")
            return paths

        depth = _clamp(getattr(graph_query, "max_depth", 2), 1, _MAX_TRAVERSAL_DEPTH, 2)
        query_type = getattr(graph_query, "query_type", QueryType.SUBGRAPH)

        # 主路径实际使用的深度由 _expand_multi_hop_paths 回填（它内部会按种子度数降级）。
        # 先置 None：避免上一次查询的残留值影响本次的"兜底是否等价"判定。
        self._last_effective_depth = None

        try:
            # 按查询类型分派到不同的遍历策略；每条策略内部自行完成种子解析与结果组装
            with self.driver.session() as session:
                if query_type == QueryType.ENTITY_RELATION:
                    paths = self._find_entity_relations(graph_query, session)
                elif query_type == QueryType.PATH_FINDING:
                    paths = self._find_shortest_paths(graph_query, session)
                elif query_type == QueryType.CLUSTERING:
                    paths = self._find_similar_entities(graph_query, session)
                else:
                    # MULTI_HOP / SUBGRAPH：以种子为起点做受限多跳扩展
                    paths = self._expand_multi_hop_paths(graph_query, session, depth)
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
            paths = []

        # 兜底：按类型分派后若为空（例如 PATH_FINDING 没解析出目标实体、聚类种子无相似项），
        # 退一步做一次通用的邻居扩展，尽量避免对用户静默返回空结果。
        # 【但 MULTI_HOP / SUBGRAPH 分支不需要这个兜底】它们在主路径上调用的就是
        # _expand_multi_hop_paths，且实际深度 <= 2 时兜底参数 min(depth,2)==实际深度，两轮调用的
        # Cypher 文本与参数【全等】（实测一次调用共发出 8 条 Cypher）；种子解析又是确定性的，
        # 变长路径集在 hub 剪枝下单调，所以第二轮必然得到同样的空结果 ——
        # 纯粹是把最贵的那条查询原样重跑一遍，耗时翻倍（实测 SUBGRAPH d=2 约 2.0x）。
        # 只有 ENTITY_RELATION / PATH_FINDING / CLUSTERING 三类分支，主路径用的是别的查询，
        # 兜底才可能真的补到结果。
        #
        # 【判据必须用"实际深度"而不是"请求深度"】_expand_multi_hop_paths 内部会按种子度数降级
        # （hub 种子 3->1，高密度种子 3->2）。用请求深度判断时，max_depth=3 + hub 种子会出现：
        # 主路径按请求 3 判为"非等价"→ 放行兜底 → 兜底以 min(3,2)=2 进入、同样被降级到 1
        # → 发出的 Cypher 与参数与主路径【逐字相同】，白跑一遍。
        # 用实际深度判断则天然覆盖：实际深度 <= 2 时，兜底的 2/1 跳与它等价（降级规则是单调的，
        # 实际深度 1 意味着种子是 hub，兜底也必然降到 1），故一律跳过。
        effective_depth = (self._last_effective_depth
                           if self._last_effective_depth is not None else depth)
        needs_generic_fallback = (
            query_type in (QueryType.ENTITY_RELATION, QueryType.PATH_FINDING, QueryType.CLUSTERING)
            or effective_depth > 2
        )
        if not paths and needs_generic_fallback:
            logger.info("按查询类型未取得路径，执行通用邻居扩展兜底")
            try:
                with self.driver.session() as session:
                    paths = self._expand_multi_hop_paths(graph_query, session, min(depth, 2))
            except Exception as e:
                logger.warning(f"通用多跳兜底失败: {e}")
        elif not paths:
            logger.info("主路径未取得路径，且兜底查询与主路径完全等价，跳过一次重复遍历")

        paths = self._dedup_paths(paths)
        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths

    def _effective_depth_for_seeds(self, seeds: List[Dict[str, Any]], depth: int) -> int:
        """按种子节点的度数决定实际可用的遍历深度。

        背景（实测）：hub 剪枝只作用于路径的中间节点，种子作为端点从未受检。而图谱里存在
        度数 511 的 CookingStep（大量同名「步骤2/步骤3」节点）与度数 1556 的 Category。
        若把这类节点当种子做 3 跳变长路径展开，Neo4j 会枚举上百万行后オ再被 LIMIT 截断
        （实测 dbHits=12,809,992、枚举 1,395,178 行、6.7-9.7s），而 depth=2 时同一查询仅 296ms。
        换言之 LIMIT 只能截断「返回」，截断不了「枚举」。

        因此这里分成两档（都是"用可接受的精度换掉不可接受的延迟"）：
          - 度数 > _HUB_DEGREE_THRESHOLD：压到 1（只取直接邻居）；
          - 度数 > _DEEP_SEED_DEGREE_LIMIT 且请求了 3 跳：压到 2。
            3 跳枚举量随种子度数相乘增长，实测度数 83 的种子 3 跳要 4.7s 却只比 2 跳多 2 个节点，
            而降级到 2 跳只要 85ms；度数 <= 12 的小种子（含全部食材种子，度数恒为 3）不受影响，
            它们才是"第三跳真能走到新内容"的那一类（实测新增 32 个节点）。
        退化只发生在高密度种子上，以菜谱/食材为种子的常规查询不受影响。
        """
        if not seeds:
            return depth
        max_degree = max(int(s.get("degree") or 0) for s in seeds)
        if max_degree > _HUB_DEGREE_THRESHOLD and depth > 1:
            logger.info(
                f"种子节点度数为 {max_degree}（> {_HUB_DEGREE_THRESHOLD}），"
                f"遍历深度由 {depth} 降级为 1 以避免变长路径枚举爆炸"
            )
            return 1
        if depth > 2 and max_degree > _DEEP_SEED_DEGREE_LIMIT:
            logger.info(
                f"种子节点度数为 {max_degree}（> {_DEEP_SEED_DEGREE_LIMIT}），"
                f"遍历深度由 {depth} 降级为 2：该种子的第 3 跳只是在已覆盖邻域内绕圈，"
                f"枚举量却随度数相乘增长"
            )
            return 2
        return depth

    def _select_expansion_seeds(self, seeds: List[Dict[str, Any]], depth: int) -> List[str]:
        """挑选真正参与展开的种子 id。

        Cypher 里用的是同一条变长路径模式，UNWIND 多个种子意味着【每个种子各自完整枚举一遍】，
        而 LIMIT 作用在并集上、只能截断返回行数，截断不了枚举量。实测：
        1 个菜谱种子 145ms / 5 个种子 4,394ms，而两者返回的都是那 60 条路径。
        所以深度 >= 2 时限制种子个数（浅遍历便宜，可全用）。

        seeds 已由 _resolve_seed_nodes 按 (匹配级别 DESC, 度数 DESC) 排好序，
        因此取前 N 个即"最匹配且信息量最大"的那几个。
        """
        if not seeds:
            return []
        limit = len(seeds) if depth <= 1 else _MAX_DEEP_SEED_COUNT
        selected = [s["id"] for s in seeds[:limit] if s.get("id")]
        if len(seeds) > len(selected):
            logger.info(f"深遍历种子数由 {len(seeds)} 收敛为 {len(selected)}（其余种子成本线性叠加但结果高度重叠）")
        return selected

    def _expand_pairing_paths(self, graph_query: GraphQuery, session,
                              seeds: List[Dict[str, Any]],
                              target_keywords: List[str],
                              budget: int) -> List[GraphPath]:
        """「搭配共现」通道：用"经由菜谱共现"的两跳结构找出真正的搭配对象。

        为什么需要单独一条通道：
        变长路径 `(source)-[*1..N]-(target)` 不区分"经由什么连接"，而"X 配什么 Y"问句的语义
        恰恰是【X 和 Y 出现在同一个菜谱里】。用变长路径回答这类问题有两处结构性错配：

        1. 路径预算被种子数瓜分。`_MAX_DEEP_SEED_COUNT` 把参与深遍历的种子收口到 2 个，
           而每个食材种子只能走到"自己所在的那个菜谱"，于是 60 条路径预算全落在 1~2 个菜谱上。
           实测「鸡肉配什么蔬菜好？」只返回 2 条路径，全部来自「麻辣香锅」这一个菜谱 ——
           而图谱里与鸡肉共现的蔬菜有 27 种。
        2. 目标侧必须靠"路径终点自己匹配关键词"来收敛。终点是食材时要靠 category 兜住，
           而"最短路径优先"的截断又会先填满 1 跳路径（食材→菜谱、食材→分类），
           真正有信息量的"食材→菜谱→食材"反而排在后面。

        本通道把这两点一次性解决：
          * 源侧按【名字集合】锚定（用户原词 + 同族别名 + 已解析出的种子名），
            因此同名反规范化副本【全部】参与，不受种子数上限影响；
          * 目标侧在 Cypher 内就约束住（名字/分类/所属 Category 节点），
            不依赖事后剪枝；
          * 聚合与排序交给 Neo4j：按"共现菜谱数"降序，
            即"最常和鸡肉一起出现的蔬菜"排在最前 —— 这正是"配什么"问句想要的答案顺序。

        成本（实测，12 个鸡肉食材名 -> 蔬菜）：6~34ms、返回 27 个搭配；
        同一场景下 3 跳变长路径要枚举 3963 行、680ms。

        Args:
            graph_query: 查询对象（只读 source_entities，不修改）
            session: 已建立的 Neo4j 会话
            seeds: _resolve_seed_nodes 解析出的种子（用于补充源侧名字）
            target_keywords: 已规范化的目标关键词（非空，由调用侧保证）
            budget: 本通道最多产出的搭配路径数

        Returns:
            GraphPath 列表（path_type='multi_hop_pairing'，已按共现强度降序、确定性排序）
        """
        budget = max(1, int(budget))
        raw_names = _dedup_preserve_order([
            self._normalize_entity_name(e) for e in (getattr(graph_query, "source_entities", None) or [])
            if self._normalize_entity_name(e)
        ])
        if not raw_names:
            return []

        # 源侧名字集合 = 用户原词 + 同族别名 + 已解析出的食材种子名
        # （三者取并集而不是只取其一：原词保证"用户说的词"一定在内，
        #   别名保证"图谱里的同族说法"被覆盖，种子名保证"解析结果"不会因为别名被截断而丢失）
        source_names = _dedup_preserve_order(
            raw_names
            + self._expand_alias_names(raw_names)
            + [str(s.get("name") or "") for s in (seeds or [])
               if "Ingredient" in (s.get("labels") or []) and s.get("name")]
        )
        source_names = [n for n in source_names if n]
        if not source_names:
            return []

        # 见证（witness）取多条再在 Python 侧取确定性的一条：
        # Cypher 的 collect 顺序由执行计划决定，不保证稳定；而检索结果必须可复现。
        cypher = """
        MATCH (i:Ingredient)<-[r1:REQUIRES]-(r:Recipe)-[r2:REQUIRES]->(v:Ingredient)
        WHERE i.name IN $source_names
          AND i.nodeId IS NOT NULL AND v.nodeId IS NOT NULL AND r.nodeId IS NOT NULL
          AND NOT coalesce(i.nodeId, '') IN $placeholder_ids
          AND NOT coalesce(v.nodeId, '') IN $placeholder_ids
          AND NOT coalesce(r.nodeId, '') IN $placeholder_ids
          AND i.nodeId <> v.nodeId
          AND any(k IN $target_keywords WHERE
                    toLower(coalesce(v.name, '')) CONTAINS k
                 OR toLower(coalesce(v.category, '')) CONTAINS k
                 // 图谱不变量（实测 2905/2905）：v.category 与它所属 Category 节点的名字完全一致，
                 // 所以上面一条已足够。这里保留类别边兜底，是为了"未来导入的食材缺 category 属性"
                 // 时仍能命中；实测加上它耗时无可见变化（21ms vs 25ms，均在噪声内）。
                 OR EXISTS { MATCH (v)-[:BELONGS_TO_CATEGORY]->(c:Category)
                             WHERE toLower(coalesce(c.name, '')) CONTAINS k })
        // 按【名字】聚合，而不是按节点：食材节点按菜谱反规范化复制，
        // 「姜」在图谱里有 64 个节点，按节点聚合会把 6 个菜谱的共现拆成 6 条 support=1 的记录。
        WITH v.name AS v_name, v.category AS v_cat,
             count(DISTINCT r) AS support,
             count(DISTINCT i.name) AS src_name_count,
             collect(DISTINCT {i: i, r: r, r1: r1, r2: r2, v: v})[0..$witness] AS witnesses
        ORDER BY support DESC, src_name_count DESC, v_name ASC
        LIMIT $limit
        UNWIND witnesses AS w
        RETURN v_name, v_cat, support, src_name_count,
               w.i AS i_node, w.r AS r_node, w.r1 AS r1_rel, w.r2 AS r2_rel,
               // 目标节点【随见证一起取回】而不是用 r2.end_node 现取：
               // 驱动对关系的 end_node 是按需装配的，若该节点未随记录返回，
               // 在结果集消费完后访问可能拿不到（甚至报错），而这里对失败的容错是「静默跳过」，
               // 一旦发生就是整条通道全部丢结果却毫无痕迹。
               w.v AS v_node
        """

        rows: List[Dict[str, Any]] = []
        try:
            rows = [dict(record) for record in session.run(cypher, {
                "source_names": source_names,
                "target_keywords": list(target_keywords),
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "witness": _PAIRING_WITNESS_LIMIT,
                "limit": budget * _PAIRING_ROW_FACTOR,
            })]
        except Exception as e:
            logger.error(f"搭配共现查询失败: {e}")
            return []

        if not rows:
            return []

        # 按 (搭配对象名, 类目) 归并见证行，并选出唯一一条见证三元组（按 nodeId 取最小，保证可复现）
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("v_name") or ""), str(row.get("v_cat") or ""))
            entry = grouped.setdefault(key, {
                "support": int(row.get("support") or 0),
                "src_name_count": int(row.get("src_name_count") or 0),
                "witnesses": [],
            })
            i_node, r_node, v_node = row.get("i_node"), row.get("r_node"), row.get("v_node")
            if i_node is None or r_node is None or v_node is None:
                continue
            entry["witnesses"].append({
                "i_id": str(i_node.get("nodeId") or ""),
                "r_id": str(r_node.get("nodeId") or ""),
                "i_node": i_node, "r_node": r_node, "v_node": v_node,
                "r1": row.get("r1_rel"), "r2": row.get("r2_rel"),
            })

        paths: List[GraphPath] = []
        for (v_name, _v_cat), entry in grouped.items():
            witnesses = [w for w in entry["witnesses"] if w["r1"] is not None and w["r2"] is not None]
            if not witnesses:
                continue
            witness = sorted(witnesses, key=lambda w: (w["r_id"], w["i_id"]))[0]

            # 见证三元组 (源食材, 菜谱, 目标食材) 三个节点都直接来自同一条记录，
            # 因此「描述出来的路径」与「support 统计所依据的那条路径」必然一致，
            # 不会出现"统计来自 A 菜谱、展示却用了 B 菜谱"的错配。
            nodes = [
                self._node_to_dict(witness["i_node"]),
                self._node_to_dict(witness["r_node"]),
                self._node_to_dict(witness["v_node"]),
            ]
            if not all(nodes):
                continue

            rels = [self._rel_to_dict(witness["r1"]), self._rel_to_dict(witness["r2"])]
            path = self._build_graph_path(nodes=nodes, rels=rels, path_type="multi_hop_pairing")
            if not path:
                continue

            # 共现强度写进第二条关系（食材-菜谱-食材 里的最后一条）：
            # 它是"这条路径为什么值得信"的额外证据，_build_path_description 会把它渲染进正文，
            # 让生成模型能说出"在 N 个菜谱里和鸡肉一起出现"这种可核查的表述。
            rels[-1]["pairing_support"] = entry["support"]
            rels[-1]["pairing_source_names"] = entry["src_name_count"]
            paths.append(path)

        # 度数必须在 _score_path 之前补齐：_score_path 的「节点信息量」分量直接读 n["degree"]，
        # 批量补度数一次往返即可（见 _attach_degrees 的性能说明），不要逐节点查。
        self._attach_degrees([n for p in paths for n in p.nodes])

        for path in paths:
            path.relevance_score = round(
                min(self._score_path(path) + self._pairing_support_bonus(path), 0.99), 6
            )

        # 二次排序：Cypher 已按 (support, src_name_count, v_name) 排好，
        # 这里再按同一口径显式排序一次，避免归并（dict 插入序）与 Cypher 顺序不一致时的歧义。
        paths.sort(key=lambda p: (-float(p.relationships[-1].get("pairing_support") or 0),
                                  -float(p.relationships[-1].get("pairing_source_names") or 0),
                                  str(p.nodes[-1].get("name") or "")))
        logger.info(
            f"搭配共现通道：源名字 {len(source_names)} 个 -> 命中 {len(paths)} 个搭配"
            f"（top: {'、'.join(str(p.nodes[-1].get('name')) for p in paths[:5])}）"
        )
        return paths[:budget]

    def _expand_multi_hop_paths(self, graph_query: GraphQuery, session,
                                depth: int) -> List[GraphPath]:
        """受限多跳扩展：本模块多跳检索的主力实现。

        为什么不用「一次性把 N 跳全查出来」：
        实测从单个菜谱出发的变长路径 1 跳 35 条、2 跳 698 条、3 跳【127214 条】，
        因此这里做了五层收口：
          1. 深度钳制在 [_MAX_TRAVERSAL_DEPTH] 以内，并以字面量内插（Neo4j 不允许参数化上界）；
          2. Cypher 内过滤标签白名单与占位脏节点，把统计/层次节点排除在路径之外；
          3. Cypher 内 LIMIT 控制单次返回规模，再在 Python 侧做 hub 剪枝（度数超阈值的中间节点剔除），
             因为把 COUNT{(n)--()} 放进路径匹配里逐节点求值代价过高，不如取回后批量补度数再剪；
          4. 【种子侧也要收口】：hub 剪枝原本只作用于路径的"中间节点"，而种子是端点、从不被检查。
             实测把度数 511 的 CookingStep '步骤3'（图谱里存在大量同名步骤节点）当种子做 3 跳展开，
             VarLengthExpand 会枚举 1,395,178 行、耗时 6.7-9.7s，而 LIMIT 60 只能截断【返回】不能截断【枚举】。
             所以这里按种子的最大度数主动降级深度（见 _effective_depth_for_seeds），
             并限制参与深展开的种子个数（见 _MAX_DEEP_SEED_COUNT）——多种子是线性叠加的枚举成本，
             实测 5 个种子做 3 跳展开要 4.4s，而结果同样是那 60 条。
          5. 目标实体约束走「名称/标签/分类/菜系」多字段语义匹配，而不是错误的 labels() 直比。
          6. 【本方法有两条通道，共用一份路径预算】：
             - 搭配共现通道（_expand_pairing_paths）：把"X 配什么 Y"这类问句解释为
               「X 与 Y 经由同一个菜谱共现」，直接用 食材-菜谱-食材 的结构化两跳回答，
               并按共现菜谱数排序。它按【名字】锚定源侧，因此同名反规范化副本全部参与，
               不受 _MAX_DEEP_SEED_COUNT 限制。
             - 通用变长路径通道（本方法主体）：回答"通过什么连到"这类开放问句。
             预算分配是【搭配通道优先】：通用通道的 LIMIT 是"最短路径优先"，
             1 跳路径（食材→菜谱）会把 LIMIT 全部吃掉，2 跳的"食材→菜谱→食材"根本排不进来。
             实测「鸡肉配什么蔬菜好？」改造前只返回 2 条路径（都来自同一个菜谱），
             搭配通道介入后返回 27 种蔬菜。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=5, session=session)
        if not seeds:
            logger.info(f"多跳扩展未解析出种子节点: {graph_query.source_entities}")
            return []

        # 种子侧的两道收口（顺序重要：先按度数决定深度，再按深度决定用几个种子）
        depth = self._effective_depth_for_seeds(seeds, depth)
        # 回填"本次实际使用的深度"：调用方（multi_hop_traversal）靠它判断兜底查询是否与主路径等价
        self._last_effective_depth = depth
        seed_ids = self._select_expansion_seeds(seeds, depth)
        target_keywords = self._normalize_target_keywords(graph_query.target_entities)
        rel_types = [r for r in (graph_query.relation_types or []) if r in _RELATION_WEIGHTS]
        effective_rel_types = rel_types or list(_RELATION_WEIGHTS.keys())

        # 【搭配共现通道】先跑，并且它的产出【先从总预算里扣除】（见预算分配的注释）。
        # 触发条件三选全满足才跑：
        #   - depth >= 2：1 跳只能答"这个食材属于哪个类目"，答不了"和谁搭配"；
        #   - target_keywords 非空：搭配通道的目标侧收敛完全依赖关键词，没有关键词就退化成全量共现；
        #   - 种子里有 Ingredient：源侧要按名字锚定，全是菜谱/步骤种子时这条通道无意义。
        pairing_paths: List[GraphPath] = []
        if depth >= 2 and target_keywords and any(
                "Ingredient" in (s.get("labels") or []) for s in seeds):
            pairing_paths = self._expand_pairing_paths(
                graph_query, session, seeds, target_keywords,
                budget=max(_PAIRING_MIN_RESULTS,
                           int(_MAX_PATH_RESULTS * _PAIRING_BUDGET_RATIO)),
            )
            if len(pairing_paths) < _PAIRING_MIN_RESULTS:
                # 命中太少说明这条问句不是"搭配"型（或图谱里确实没有共现），
                # 与其塞几条弱证据进去污染排序，不如整条通道退场、把预算还给通用通道。
                logger.info(f"搭配共现命中 {len(pairing_paths)} 条，低于阈值 {_PAIRING_MIN_RESULTS}，放弃该通道")
                pairing_paths = []

        # 预算分配：搭配路径【优先占用】总预算，通用变长路径只拿剩下的部分。
        # 为什么必须让路：通用通道的 LIMIT 是"最短路径优先"（ORDER BY path_len ASC），
        # 1 跳路径（食材->菜谱）会把 LIMIT 全部吃掉，2 跳的"食材->菜谱->食材"根本排不进来 ——
        # 这正是改造前「鸡肉配什么蔬菜」只返回 2 条路径的直接原因。
        # 剩额为 0 时干脆不跑通用查询：省掉一次 680ms 级的枚举，且不会产出任何会被丢弃的结果。
        generic_limit = max(_MAX_PATH_RESULTS - len(pairing_paths), 0)
        if generic_limit == 0:
            logger.info(f"搭配共现已占满路径预算（{len(pairing_paths)} 条），跳过通用变长路径查询")
            pairing_paths.sort(key=lambda p: p.relevance_score, reverse=True)
            return pairing_paths

        # 深度以字面量内插：Neo4j 明确拒绝 `[*1..$depth]` 这种参数化上界，
        # 而 depth 已由 _clamp 与 _effective_depth_for_seeds 保证是 [1,3] 内的整数，不存在注入风险。
        cypher = f"""
        MATCH (source) WHERE source.nodeId IN $seed_ids
        MATCH path = (source)-[*1..{depth}]-(target)
        WHERE NOT target = source
          AND all(x IN nodes(path) WHERE ({_PATH_LABEL_FILTER_X}))
          AND all(x IN nodes(path) WHERE NOT coalesce(x.nodeId, '') IN $placeholder_ids)
          AND any(r IN relationships(path) WHERE type(r) IN $rel_types)
          AND ($target_keywords = []
               OR any(k IN $target_keywords WHERE
                    toLower(coalesce(target.name, '')) CONTAINS k
                    OR toLower(coalesce(target.category, '')) CONTAINS k
                    OR toLower(coalesce(target.cuisineType, '')) CONTAINS k
                    OR any(lbl IN labels(target) WHERE toLower(lbl) CONTAINS k)))
        WITH path, length(path) AS path_len, nodes(path) AS pnodes, relationships(path) AS prels
        ORDER BY path_len ASC
        LIMIT $limit
        RETURN pnodes, prels, path_len
        """

        rows = []
        try:
            rows = [dict(record) for record in session.run(cypher, {
                "seed_ids": seed_ids,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "rel_types": effective_rel_types,
                "target_keywords": target_keywords,
                "limit": generic_limit,
            })]
        except Exception as e:
            # 通用通道失败不该连带丢掉已经拿到的搭配路径：两者是独立证据源，失败隔离。
            logger.error(f"多跳路径查询失败: {e}")
            return pairing_paths

        paths: List[GraphPath] = []
        for row in rows:
            path = self._build_graph_path(
                nodes=[self._node_to_dict(n) for n in (row.get("pnodes") or [])],
                rels=[self._rel_to_dict(r) for r in (row.get("prels") or [])],
                path_type="multi_hop",
            )
            if path:
                paths.append(path)

        if not paths:
            return pairing_paths

        # 统一补齐度数后再剪枝：hub 剪枝与结构评分都依赖度数
        all_nodes = [n for p in paths for n in p.nodes]
        self._attach_degrees(all_nodes)

        pruned: List[GraphPath] = []
        dropped_hub = 0
        for path in paths:
            # 中间节点（非起点、非终点）若为 hub 则整条路径丢弃：
            # 实测 Category '调料'(1556) 这类 hub 会把任意两个菜谱连起来，属于噪声而非知识
            intermediates = path.nodes[1:-1]
            if any(self._is_hub(n) for n in intermediates):
                dropped_hub += 1
                continue
            path.relevance_score = self._score_path(path)
            pruned.append(path)

        if dropped_hub:
            logger.info(f"hub 剪枝丢弃 {dropped_hub} 条路径（中间节点度数 > {_HUB_DEGREE_THRESHOLD}）")

        # 合并两条通道的结果。
        # 搭配路径【不过 hub 剪枝】：它的中间节点按构造就是 Recipe（共现的见证者），
        # 剪掉它等于剪掉这条通道的全部语义；而且 Recipe 度数实测 3~30，远低于阈值 120，
        # 真正需要防的 hub（Category '调料' 1556、'蔬菜' 605）根本不会出现在 食材-菜谱-食材 里。
        # 排序：按 relevance_score 降序，同分时 Python 稳定排序保留输入顺序 ——
        # 这里把搭配路径排在前面，所以同分优先保留搭配证据（它是本问句的直接答案）。
        merged = pairing_paths + pruned
        merged.sort(key=lambda p: p.relevance_score, reverse=True)
        return merged[:_MAX_PATH_RESULTS]

    def _score_path(self, path: GraphPath) -> float:
        """给一条图路径打相关性分。

        三个分量（权重合计 1.0）：
          - 关系语义强度 (0.5)：路径上各关系类型权重的均值，体现「这条路径说的话是否靠谱」；
          - 路径紧凑度 (0.3)：1/(1+跳数)，越短越可信（间接关联噪声更大）；
          - 节点信息量 (0.2)：路径节点度数的归一化均值，避免全是孤立叶子节点的路径。
        """
        if not path.relationships:
            rel_strength = _DEFAULT_RELATION_WEIGHT
        else:
            weights = [
                _RELATION_WEIGHTS.get(str(r.get("type") or ""), _DEFAULT_RELATION_WEIGHT)
                for r in path.relationships
            ]
            rel_strength = sum(weights) / float(len(weights))

        compactness = 1.0 / (1.0 + max(int(path.path_length or 1), 1))

        if path.nodes:
            degree_score = sum(min(int(n.get("degree") or 0) / 50.0, 1.0) for n in path.nodes) / float(len(path.nodes))
        else:
            degree_score = 0.0

        score = 0.5 * rel_strength + 0.3 * compactness + 0.2 * degree_score
        return round(float(score), 6)

    def _pairing_support(self, path: GraphPath) -> Tuple[int, int]:
        """读取路径上的搭配共现证据。

        证据写在【关系字典】上（见 _expand_pairing_paths）：pairing_support = 共现菜谱数，
        pairing_source_names = 该搭配覆盖的同族食材名数。
        一条路径上可能有多条关系带该字段，取最大值；任何异常（类型不符）都按"无证据"处理，
        因为这里只是加分/展示的依据，不值得为脏数据中断检索。

        Returns:
            (共现菜谱数, 同族食材名数)；无证据时为 (0, 0)
        """
        support, src_count = 0, 0
        for rel in (path.relationships or []):
            try:
                support = max(support, int(rel.get("pairing_support") or 0))
                src_count = max(src_count, int(rel.get("pairing_source_names") or 0))
            except (TypeError, ValueError):
                continue
        return support, src_count

    def _pairing_support_bonus(self, path: GraphPath) -> float:
        """把搭配共现强度折成一个【有界】的结构证据加成，取值 [0, _PAIRING_SUPPORT_BONUS]。

        为什么是"结构证据"而不是词法证据：结构完全相同的 食材-菜谱-食材 路径（2 跳、两条 REQUIRES）
        在 _score_path / _path_struct_score 里拿到的是【完全相同】的分数，
        唯一能区分"鸡肉配姜很常见"和"鸡肉配欧芹只出现一次"的信号就是共现菜谱数，
        它衡量的是图谱里这条关联的强度，属于结构侧。
        上限存在的原因：避免高频搭配压过一切其它证据，也让同分时退回稳定排序的输入顺序。
        """
        support, _ = self._pairing_support(path)
        if support <= 0:
            return 0.0
        return _PAIRING_SUPPORT_BONUS * min(float(support) / _PAIRING_SUPPORT_SATURATION, 1.0)

    def _build_graph_path(self, nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]],
                          path_type: str, path_length: Optional[int] = None) -> Optional[GraphPath]:
        """把节点/关系字典组装成 GraphPath，并做统一的脏数据过滤。

        返回 None 表示这条路径不该进入结果集（节点为空、或混入了统计/占位脏节点）。
        """
        # 过滤掉统计/层次节点与占位节点；只要混入一个，整条路径即为脏路径
        clean_nodes = [n for n in nodes if self._is_meaningful_node(n)]
        if not clean_nodes or len(clean_nodes) != len(nodes):
            return None

        if len(clean_nodes) == 1 and not rels:
            # 单节点无关系无法构成路径
            return None

        return GraphPath(
            nodes=clean_nodes,
            relationships=list(rels or []),
            path_length=int(path_length if path_length is not None else max(len(clean_nodes) - 1, 1)),
            relevance_score=0.0,   # 由 _score_path 统一回填，保证评分口径一致
            path_type=path_type,
        )

    def _dedup_paths(self, paths: List[GraphPath]) -> List[GraphPath]:
        """按「节点序列」去重，保留相关性最高的那条。

        Cypher 的无向多跳匹配会对同一批节点产出方向/顺序不同的重复路径，必须去重否则 top_k 会被挤占。
        """
        best: Dict[Tuple[str, ...], GraphPath] = {}
        for path in paths:
            if not path or not path.nodes:
                continue
            key = tuple(str(n.get("id") or n.get("name") or "") for n in path.nodes)
            if key not in best or path.relevance_score > best[key].relevance_score:
                best[key] = path
        ordered = sorted(best.values(), key=lambda p: p.relevance_score, reverse=True)
        return ordered[:_MAX_PATH_RESULTS]

    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系（ENTITY_RELATION 分支的真实实现）。

        策略是「先直连、再两跳」：
          1. 先找种子实体与其邻居之间的直连关系；给了目标实体就按名称/分类做约束；
          2. 直连为空且给定目标实体时，放宽到两跳（A -r1- mid -r2- B），
             这对应「A 和 B 之间通过什么关联」这类问法。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=5, session=session)
        if not seeds:
            return []

        seed_ids = [s["id"] for s in seeds]
        targets = self._resolve_seed_nodes(graph_query.target_entities, limit=5, session=session) \
            if graph_query.target_entities else []
        target_ids = [t["id"] for t in targets]
        target_keywords = self._normalize_target_keywords(graph_query.target_entities)

        paths: List[GraphPath] = []

        # 第一级：直连关系
        # 【为什么必须带 ORDER BY】原先只有裸 LIMIT $limit，截断点由存储层决定：
        # 种子度数大（如 调料 1556）时候选行远超 LIMIT，"留下哪 60 条"在不同运行之间
        # 都不稳定 —— 实测同一查询两次运行可能拿到不同的邻居集合，缓存、评测、排障
        # 全都对不上账。排序键选「邻居度数 ASC」有两重作用：
        #   1) 让截断确定可复现（并列再按 nodeId 兜底）；
        #   2) 优先保留【更具体】的邻居。度数低的节点（一个食材、一个步骤）承载的语义
        #      比度数 1556 的分类节点强得多，等于把 LIMIT 花在信息量最高的那一端。
        direct_cypher = """
        MATCH (src) WHERE src.nodeId IN $seed_ids
        MATCH (tgt)-[r]-(src)
        WHERE (tgt:Recipe OR tgt:Ingredient OR tgt:CookingStep OR tgt:Category)
          AND NOT coalesce(tgt.nodeId, '') IN $placeholder_ids
          AND ($target_ids = [] OR tgt.nodeId IN $target_ids)
          AND ($target_keywords = []
               OR any(k IN $target_keywords WHERE
                    toLower(coalesce(tgt.name, '')) CONTAINS k
                    OR toLower(coalesce(tgt.category, '')) CONTAINS k))
        WITH src, tgt, r, COUNT { (tgt)--() } AS tgt_degree
        RETURN src, tgt, r
        ORDER BY tgt_degree ASC, coalesce(tgt.nodeId, ''), type(r)
        LIMIT $limit
        """
        try:
            for record in session.run(direct_cypher, {
                "seed_ids": seed_ids,
                "target_ids": target_ids,
                "target_keywords": target_keywords,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "limit": _MAX_PATH_RESULTS,
            }):
                path = self._build_graph_path(
                    nodes=[self._node_to_dict(record["src"]), self._node_to_dict(record["tgt"])],
                    rels=[self._rel_to_dict(record["r"])],
                    path_type="entity_relation",
                    path_length=1,
                )
                if path:
                    paths.append(path)
        except Exception as e:
            logger.error(f"实体直连关系查询失败: {e}")

        # 第二级：仅当「指定了目标实体但直连为空」时才放宽到两跳，避免无谓的性能开销
        if not paths and target_keywords:
            # 【两跳必须做 hub 剪枝】这是本方法唯一的兜底出口，也是最容易产出"假捷径"的地方：
            # 中间节点只要是 Category 这类超高度数的分类节点，任意两个同分类实体之间都会
            # "接通"。实测「生抽 vs 蚝油」：直连 0 行转入本查询，匹配集共 221 行、
            # 220 行都经过 Category '调料'（唯一个节点、度数 1556），Cypher LIMIT 60
            # 截断后返回的 60 条里有 59 条是
            # 生抽 -[BELONGS_TO_CATEGORY]-> 调料(1556) <-[BELONGS_TO_CATEGORY]- 蚝油。
            # 且 _dedup_paths 一条都合并不掉 —— 它的键是节点序列，而目标侧因为食材
            # 按菜谱反规范化复制而生成了 60 个同名不同 nodeId 的节点，60 条路径的键互不相同。
            # 结果就是 top_k 被同一条语义的重复路径彻底挤占。
            # 剪枝口径与 _expand_multi_hop_paths / _find_shortest_paths 保持一致
            #（中间节点度数 > _HUB_DEGREE_THRESHOLD 即视为 hub，不承担中转语义）。
            # 注意：剪枝的只是"中间节点"，端点不做限制，所以真实的两跳关系不受影响。
            two_hop_cypher = """
            MATCH (src) WHERE src.nodeId IN $seed_ids
            MATCH (src)-[r1]-(mid)-[r2]-(tgt)
            WHERE tgt <> src
              AND (mid:Recipe OR mid:Ingredient OR mid:CookingStep OR mid:Category)
              AND (tgt:Recipe OR tgt:Ingredient OR tgt:CookingStep OR tgt:Category)
              AND NOT coalesce(mid.nodeId, '') IN $placeholder_ids
              AND NOT coalesce(tgt.nodeId, '') IN $placeholder_ids
              AND any(k IN $target_keywords WHERE
                    toLower(coalesce(tgt.name, '')) CONTAINS k
                    OR toLower(coalesce(tgt.category, '')) CONTAINS k)
            WITH src, mid, tgt, r1, r2, COUNT { (mid)--() } AS mid_degree
            WHERE mid_degree <= $hub_degree_threshold
            RETURN src, mid, tgt, r1, r2
            ORDER BY mid_degree ASC, coalesce(tgt.nodeId, ''), coalesce(mid.nodeId, '')
            LIMIT $limit
            """
            try:
                for record in session.run(two_hop_cypher, {
                    "seed_ids": seed_ids,
                    "target_keywords": target_keywords,
                    "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                    "hub_degree_threshold": _HUB_DEGREE_THRESHOLD,
                    "limit": _MAX_PATH_RESULTS,
                }):
                    path = self._build_graph_path(
                        nodes=[self._node_to_dict(record["src"]),
                               self._node_to_dict(record["mid"]),
                               self._node_to_dict(record["tgt"])],
                        rels=[self._rel_to_dict(record["r1"]), self._rel_to_dict(record["r2"])],
                        path_type="entity_relation",
                        path_length=2,
                    )
                    if path:
                        paths.append(path)
            except Exception as e:
                logger.error(f"实体两跳关系查询失败: {e}")

        if not paths:
            return []

        self._attach_degrees([n for p in paths for n in p.nodes])
        for path in paths:
            path.relevance_score = self._score_path(path)
        return paths[:_MAX_PATH_RESULTS]

    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径（PATH_FINDING 分支的真实实现）。

        用 Cypher 原生 shortestPath 求两端点间的最短通路，并强制：
          - 路径上所有节点都必须属于【可作中转的】标签白名单（_PATH_LABEL_FILTER_X，刻意不含
            ConceptType 与 DifficultyLevel）—— 实测不加这层过滤时，最短路径会穿过
            name 为 'Recipe' / 'Ingredient' 的层次节点，得到毫无业务意义的"路径"；
          - 排除占位脏节点（NULL 安全的 coalesce 写法，否则 Category 会被三值逻辑误杀）；
          - 上界同样只能以字面量内插（shortestPath 也不接受参数化上界）；
          - Python 侧再做一次 hub 剪枝（与多跳分支保持一致）：
            实测「宫保鸡丁 → 水煮鱼」的最短路径会返回 `宫保鸡丁 -DIFFICULTY_LEVEL-> 四星 -DIFFICULTY_LEVEL-> 水煮鱼`，
            它唯一的"知识"就是两道菜都是四星。白名单已把 DifficultyLevel 挡在外面，
            这里再加一道度数剪枝，防止修好 NULL 过滤后度数 1556 的 Category 变成新的捷径。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=3, session=session)
        if not seeds:
            return []

        targets = self._resolve_seed_nodes(graph_query.target_entities, limit=3, session=session)
        if not targets:
            logger.info(f"最短路径未解析出目标实体: {graph_query.target_entities}")
            return []

        seed_ids = [s["id"] for s in seeds]
        target_ids = [t["id"] for t in targets]

        # 上界钳制到 [2,6]：太短找不到通路，太长则 shortestPath 的搜索代价陡增
        max_len = _clamp(getattr(graph_query, "max_depth", 4) + 1, 2, 6, 5)

        cypher = f"""
        MATCH (a) WHERE a.nodeId IN $seed_ids
        MATCH (b) WHERE b.nodeId IN $target_ids
        MATCH path = shortestPath((a)-[*..{max_len}]-(b))
        WHERE all(x IN nodes(path) WHERE ({_PATH_LABEL_FILTER_X}))
          AND all(x IN nodes(path) WHERE NOT coalesce(x.nodeId, '') IN $placeholder_ids)
        RETURN nodes(path) AS pnodes, relationships(path) AS prels, length(path) AS path_len
        LIMIT $limit
        """

        paths: List[GraphPath] = []
        try:
            for record in session.run(cypher, {
                "seed_ids": seed_ids,
                "target_ids": target_ids,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "limit": _MAX_PATH_RESULTS,
            }):
                path = self._build_graph_path(
                    nodes=[self._node_to_dict(n) for n in (record.get("pnodes") or [])],
                    rels=[self._rel_to_dict(r) for r in (record.get("prels") or [])],
                    path_type="shortest_path",
                    path_length=record.get("path_len"),
                )
                if path:
                    paths.append(path)
        except Exception as e:
            logger.error(f"最短路径查询失败: {e}")
            return []

        if not paths:
            return []

        self._attach_degrees([n for p in paths for n in p.nodes])

        # hub 剪枝：丢弃中间节点度数超阈值的路径（与 _expand_multi_hop_paths 口径一致）
        pruned: List[GraphPath] = []
        for path in paths:
            if any(self._is_hub(n) for n in path.nodes[1:-1]):
                logger.info("最短路径经过 hub 中间节点，已丢弃："
                            + " → ".join(str(n.get("name")) for n in path.nodes))
                continue
            path.relevance_score = self._score_path(path)
            pruned.append(path)
        return self._dedup_paths(pruned)

    def _find_similar_entities(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找相似实体（CLUSTERING 分支的真实实现）。

        图谱实测结论决定了这里的三条通路（注意：**不能**依赖共享食材，因为食材节点是按菜谱私有的）：
          1. SIMILAR 显式相似边（精确但稀疏，全库仅约 46 道菜带此边）；
          2. 同 Category 分类（最主要的跨菜谱通路，如「水产」下有 23 道菜）；
          3. 同 cuisineType 菜系（属性级同组，如「川菜」32 道）。
        三者按 1 -> 2 -> 3 的优先级取，前一条通路够用就不再往下取，避免结果被大面积泛化的菜系刷屏。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=3, session=session)
        if not seeds:
            return []

        seed_ids = [s["id"] for s in seeds]
        limit = _clamp(getattr(graph_query, "max_nodes", 30), 5, _MAX_SUBGRAPH_NODES, 30)

        queries = [
            # 通路 1：显式 SIMILAR 边
            ("""
            UNWIND $seed_ids AS sid
            MATCH (src:Recipe) WHERE src.nodeId = sid
            MATCH (src)-[r:SIMILAR]-(peer:Recipe)
            WHERE peer.nodeId <> src.nodeId
            RETURN src, peer, r, [] AS mid_nodes, [] AS mid_rels
            LIMIT $limit
            """, "cluster_similar"),
            # 通路 2：同 Category 分类
            ("""
            UNWIND $seed_ids AS sid
            MATCH (src:Recipe) WHERE src.nodeId = sid
            MATCH (src)-[r1:BELONGS_TO_CATEGORY]->(c:Category)<-[r2:BELONGS_TO_CATEGORY]-(peer:Recipe)
            WHERE peer.nodeId <> src.nodeId
            RETURN src, peer, null AS r, [c] AS mid_nodes, [r1, r2] AS mid_rels
            LIMIT $limit
            """, "cluster_category"),
            # 通路 3：同 cuisineType 菜系（属性级，无中间节点）
            ("""
            UNWIND $seed_ids AS sid
            MATCH (src:Recipe) WHERE src.nodeId = sid
            MATCH (peer:Recipe)
            WHERE peer.cuisineType = src.cuisineType
              AND peer.cuisineType IS NOT NULL
              AND peer.nodeId <> src.nodeId
            RETURN src, peer, null AS r, [] AS mid_nodes, [] AS mid_rels
            LIMIT $limit
            """, "cluster_cuisine"),
        ]

        for cypher, path_type in queries:
            paths: List[GraphPath] = []
            try:
                for record in session.run(cypher, {"seed_ids": seed_ids, "limit": limit}):
                    mid_nodes = [self._node_to_dict(n) for n in (record.get("mid_nodes") or [])]
                    mid_rels = [self._rel_to_dict(r) for r in (record.get("mid_rels") or [])]
                    rels = list(mid_rels)
                    if record.get("r") is not None:
                        rels.append(self._rel_to_dict(record["r"]))

                    path = self._build_graph_path(
                        nodes=[self._node_to_dict(record["src"])] + mid_nodes + [self._node_to_dict(record["peer"])],
                        rels=rels,
                        path_type=path_type,
                    )
                    if path:
                        paths.append(path)
            except Exception as e:
                logger.error(f"相似实体查询失败（{path_type}）: {e}")
                continue

            if paths:
                self._attach_degrees([n for p in paths for n in p.nodes])
                for path in paths:
                    path.relevance_score = self._score_path(path)
                logger.info(f"聚类命中通路 {path_type}，返回 {len(paths)} 条")
                return paths[:_MAX_PATH_RESULTS]

        return []

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """
        提取知识子图：获取实体相关的完整知识网络
        这体现了图RAG的整体性思维

        返回子图的完整性由 attached 的 graph_metrics 如实标注，调用方应以指标为准，
        不要假设"传了 max_depth 就一定能取到那么深"：
          - depth_reached：实际达到的跳数。抽取采用【两阶段】设计（种子 1 跳邻域 +
            从非 hub 邻居再扩 1 跳），因此 depth_reached 的实测上限是 2；
            即便 max_depth 传 3，也只是沿用同一套两阶段抽取（第 3 跳会让子图被 hub
            迅速拉成全图，收益远低于噪声），此时 depth_reached 仍会诚实回报 2。
          - hub_pruned：在【阶段二前沿】上因度数超阈值被剪掉的节点数。
            它只统计"本来可以继续外扩、但被判定为 hub 而不再展开"的邻居，
            因此二跳新进来的高度数节点【不计入】该值（它们不再向外扩展，无需剪枝）。
            换句话说 hub_pruned=0 不代表子图里没有高度数节点，
            只代表一跳邻域里没有需要拦截的外扩入口。

        异常处理：主路径（含两阶段查询）整体失败时，不会返回空壳，而是转入
        _fallback_subgraph_extraction 做三级真实降级，并在 metrics 里标注 degraded 来源。
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")

        if not self._is_driver_ready():
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)

        # 缓存命中：子图抽取较重（含多轮邻域扩展），同一实体组合重复提问时可直接复用
        cache_key = self._subgraph_cache_key(graph_query)
        cached = self.subgraph_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"子图命中缓存: {cache_key[:60]}")
            return cached

        depth = _clamp(getattr(graph_query, "max_depth", 2), 1, _MAX_TRAVERSAL_DEPTH, 2)
        max_nodes = _clamp(getattr(graph_query, "max_nodes", 50), _MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES, 50)

        subgraph: Optional[KnowledgeSubgraph] = None
        try:
            with self.driver.session() as session:
                subgraph = self._collect_subgraph(session, graph_query, depth, max_nodes)
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
            subgraph = None

        # 主路径失败则走真实降级（而不是返回空壳），保证调用方拿到的至少是可用证据
        if subgraph is None or not subgraph.central_nodes:
            logger.warning("子图主路径未取得结果，执行降级抽取")
            subgraph = self._fallback_subgraph_extraction(graph_query)

        # 【降级结果不入缓存】
        # 缓存键只含 query_type / 源实体 / 目标实体 / 深度 / 节点预算，不含"本次是否降级"，
        # 而命中缓存的路径既不重新抽取也不刷新 FIFO 位置。若把降级子图写进去，
        # 一次瞬时故障（例如 Neo4j 重启期间 driver 非 None 但查询全失败，
        # _collect_subgraph 内部吞掉异常返回 None）就会把 degraded=1.0 的空壳
        # 长期钉在缓存里：Neo4j 恢复后同一 (实体, 深度, max_nodes) 组合仍然命中该壳子，
        # 表现为"图库明明好了，这个实体却一直检索不到"。
        # 代价是降级期间同一查询会重复走降级路径 —— 降级本就应当尽快被上层观测到，
        # 这个代价是刻意接受的。
        if subgraph is not None and float(subgraph.graph_metrics.get("degraded") or 0.0) > 0.0:
            logger.warning(f"子图为降级结果（degraded={subgraph.graph_metrics.get('degraded')}），不写入缓存: {cache_key[:60]}")
        else:
            self._cache_subgraph(cache_key, subgraph)
        return subgraph

    def _subgraph_cache_key(self, graph_query: GraphQuery) -> str:
        """构造子图缓存键：涵盖类型、实体、深度与节点预算，任一变化都必须重新抽取。"""
        return "|".join([
            str(getattr(graph_query.query_type, "value", graph_query.query_type)),
            ",".join(sorted(graph_query.source_entities or [])),
            ",".join(sorted(graph_query.target_entities or [])),
            str(_clamp(getattr(graph_query, "max_depth", 2), 1, _MAX_TRAVERSAL_DEPTH, 2)),
            str(_clamp(getattr(graph_query, "max_nodes", 50), _MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES, 50)),
        ])

    def _cache_subgraph(self, cache_key: str, subgraph: KnowledgeSubgraph) -> None:
        """写入子图缓存并做 FIFO 淘汰，防止长时间运行内存无界增长。"""
        if subgraph is None:
            return
        if cache_key not in self.subgraph_cache:
            self._subgraph_cache_order.append(cache_key)
        self.subgraph_cache[cache_key] = subgraph
        while len(self._subgraph_cache_order) > _SUBGRAPH_CACHE_LIMIT:
            oldest = self._subgraph_cache_order.popleft()
            self.subgraph_cache.pop(oldest, None)

    def _collect_subgraph(self, session, graph_query: GraphQuery, depth: int,
                          max_nodes: int) -> Optional[KnowledgeSubgraph]:
        """两阶段抽取知识子图（内部实现，替代原先有语法/语义缺陷的单条聚合 Cypher）。

        原实现的三个问题在这里被绕开：
          - `MATCH (source)-[r*1..n]-(neighbor)` 之后 `collect(DISTINCT r)` 收集的是「关系列表的列表」，
            后续 dict(rel) 必然报错 —— 本实现改为【逐行返回 (种子, 关系, 邻居) 三元组】，在 Python 侧组装；
          - 在 collect 之后再用 WHERE 过滤聚合结果 —— 语义不成立，本实现改为在 RETURN 阶段直接 LIMIT；
          - 只调用 result.single()，多源实体时会丢结果 —— 本实现按行累积，天然支持多实体。

        阶段一：解析种子并取其 1 跳邻域（含边属性与方向）；
        阶段二：若 depth >= 2，再从度数较低的邻居（非 hub）向外扩一跳，把"知识的第二圈"补进来。
        """
        seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=5, session=session)
        if not seeds:
            return None

        seed_ids = [s["id"] for s in seeds]
        # 行数上限：每个种子平均取若干条边即可覆盖其邻域，避免种子度数过大时把结果撑爆
        row_limit = max(max_nodes * 4, 50)

        # ---------------- 阶段一：种子 + 1 跳邻域 ----------------
        hop1_cypher = """
        MATCH (seed) WHERE seed.nodeId IN $seed_ids
        MATCH (seed)-[r]-(nb)
        WHERE (nb:Recipe OR nb:Ingredient OR nb:CookingStep OR nb:Category)
          AND NOT coalesce(nb.nodeId, '') IN $placeholder_ids
        RETURN
            seed.nodeId AS seed_id, seed.name AS seed_name, labels(seed) AS seed_labels,
            properties(seed) AS seed_props, COUNT { (seed)--() } AS seed_degree,
            type(r) AS rel_type, properties(r) AS rel_props,
            startNode(r).nodeId AS rel_start, startNode(r).name AS rel_start_name,
            endNode(r).nodeId AS rel_end, endNode(r).name AS rel_end_name,
            nb.nodeId AS nb_id, nb.name AS nb_name, labels(nb) AS nb_labels,
            properties(nb) AS nb_props, COUNT { (nb)--() } AS nb_degree
        ORDER BY nb_degree DESC
        LIMIT $row_limit
        """

        hop1_rows: List[Dict[str, Any]] = []
        preloaded_seed_rows: List[Dict[str, Any]] = []

        try:
            hop1_rows = [dict(record) for record in session.run(hop1_cypher, {
                "seed_ids": seed_ids,
                "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                "row_limit": row_limit,
            })]
        except Exception as e:
            logger.error(f"子图阶段一查询失败: {e}")
            return None

        # 【多种子的行序轮转】hop1_rows 按 nb_degree DESC 全局排序，而 _assemble_from_rows
        # 是按行序贪心填充 connected_nodes（填满 max_nodes 即停），于是种子之间的度数差异
        # 会直接变成预算分配差异：实测 ['宫保鸡丁','水煮鱼'] max_nodes=50 时，
        # 水煮鱼（一跳邻居 42 个、度数普遍 58~83）把预算吃到 40，宫保鸡丁只拿到 10，
        # 多实体问题被单实体答案淹没。这里只把行序按种子轮转（每个种子内部仍保持度数降序），
        # 让贪心填充近似变成"按种子均分预算"，且每个种子的高价值邻居仍优先入选。
        # 保留全局排序不变的部分：单种子（绝大多数查询）时轮转是恒等操作，行为逐字节不变。
        # 局限：Cypher 侧的 LIMIT $row_limit 仍是全局截断，种子度数极端悬殊时截断本身
        # 就可能把某个种子整片切掉；根治需要按种子分别下发查询，代价是查询数随种子数增长。
        hop1_rows = self._interleave_rows_by_seed(hop1_rows)

        central_nodes, connected_nodes, relationships = self._assemble_from_rows(hop1_rows, max_nodes)

        # 种子若因度数排序被挤掉，单独补一次轻量查询，保证 central_nodes 一定有内容
        # 用身份键比对：种子本身都来自 entity_cache（必有 nodeId），此处只是保持与
        # 子图内部其他集合运算一致的标识口径，避免出现两套 id 语义
        existing_ids = {self._node_identity_key(n) for n in central_nodes}
        missing_seed_ids = [sid for sid in seed_ids if sid not in existing_ids]
        if missing_seed_ids:
            try:
                preloaded_seed_rows = [dict(record) for record in session.run(
                    """
                    MATCH (n) WHERE n.nodeId IN $ids
                    RETURN n.nodeId AS seed_id, n.name AS seed_name, labels(n) AS seed_labels,
                           properties(n) AS seed_props, COUNT { (n)--() } AS seed_degree
                    """,
                    {"ids": missing_seed_ids},
                )]
            except Exception as e:
                logger.warning(f"补查种子节点失败: {e}")

            for row in preloaded_seed_rows:
                node = {
                    "id": str(row.get("seed_id") or ""),
                    "node_id": str(row.get("seed_id") or ""),
                    "name": str(row.get("seed_name") or ""),
                    "labels": list(row.get("seed_labels") or []),
                    "label": (list(row.get("seed_labels") or []) or [""])[0],
                    "properties": dict(row.get("seed_props") or {}),
                    "degree": int(row.get("seed_degree") or 0),
                }
                if node["id"] and self._is_meaningful_node(node):
                    central_nodes.append(node)

        # 【中心节点 / 邻居节点的跨列表收口】
        # central_nodes 与 connected_nodes 各自用一套去重集合（seen_central / seen_connected），
        # 两套互不相通，因此当一个被解析出来的种子同时又是另一个种子的邻居时，
        # 同一个节点会同时留在两个列表里。后果是 node_count = len(central)+len(connected)
        # 重复计数，进而把 density = 2E/(N(N-1)) 压低、把 avg_degree 抬高，
        # 并把这个虚高的节点数通过 _subgraph_struct_score 的 coverage 项影响排序、
        # 写进喂给 LLM 的子图描述文本。
        # 触发条件不是"传了 2~3 个实体"，而是"解析出的种子集合内部存在直接相邻的一对"：
        # 单个实体也能触发 —— _resolve_seed_nodes(['红烧鱼']) 会同时返回 红烧鱼 与 红烧鱼头，
        # 二者之间有 SIMILAR 边。
        # 收口口径：已作为中心节点的节点不再计入邻居列表（中心节点的语义优先级更高），
        # 但它的关系【全部保留】，只去掉重复计数。
        central_keys = {self._node_identity_key(n) for n in central_nodes}
        central_keys.discard("")
        if central_keys:
            before = len(connected_nodes)
            connected_nodes = [n for n in connected_nodes
                               if self._node_identity_key(n) not in central_keys]
            if len(connected_nodes) != before:
                logger.info(f"中心节点与邻居节点重叠收口：{before - len(connected_nodes)} 个种子节点不再重复计入邻居")

        # ---------------- 阶段二：从非 hub 邻居再扩一跳 ----------------
        depth_reached = 1
        hub_pruned = 0
        if depth >= 2:
            # current_ids 用「身份键」而非裸 nodeId：无 nodeId 的节点（Category 等）用裸 nodeId
            # 会全部塌缩成空串，导致阶段二的 exclude 把不同节点误判成同一个，
            # 已有的 Category 邻居也无法被正确识别为"已收录"。
            current_ids = {self._node_identity_key(n) for n in central_nodes + connected_nodes}
            current_ids.discard("")
            # 只从度数较低的邻居继续扩展：hub 邻居一旦展开会把整个图谱拉进来
            frontier = [n for n in connected_nodes if not self._is_hub(n)]
            hub_pruned = len(connected_nodes) - len(frontier)
            frontier = frontier[:max(1, max_nodes // 2)]
            # 前沿节点分两路下发：有 nodeId 的走 nodeId；无 nodeId 的（Category / ConceptType 等，
            # 实测 nodeId 全为 NULL）只能按 name 匹配。早期实现只传 nodeId，
            # 这类节点在阶段二【永远无法被再次展开】，荤菜 这种"近 hub"主题节点
            # 下面的第二圈知识会整片静默丢失。
            frontier_ids = [str(n.get("id") or "") for n in frontier if n.get("id")]
            frontier_names = [str(n.get("name") or "") for n in frontier if not n.get("id")]
            frontier_names = [nm for nm in frontier_names if nm]

            if frontier_ids or frontier_names:
                # 前沿条件按需拼装：只有当确实存在"无 nodeId 的前沿节点"时才引入 OR。
                # 实测本图上 OR 分支会让查询计划无法纯走 nodeId 索引
                # （同等工作量 59ms -> 96ms），而绝大多数种子（Recipe）的邻居全是
                # Ingredient / CookingStep，都带 nodeId，走不到 OR 那一支。
                # `f.nodeId IS NULL` 这个限定不能省：Category 与 RecipeCategory 存在大量同名节点
                # （水产 / 荤菜 / 调料 …），少了它就会把带 nodeId 的 RecipeCategory 也一并命中。
                if frontier_names:
                    frontier_where = ("f.nodeId IN $frontier_ids\n"
                                      "                   OR (f.nodeId IS NULL AND f.name IN $frontier_names)")
                else:
                    frontier_where = "f.nodeId IN $frontier_ids"
                hop2_cypher = f"""
                MATCH (f)
                WHERE {frontier_where}
                MATCH (f)-[r]-(nb)
                WHERE (nb:Recipe OR nb:Ingredient OR nb:CookingStep OR nb:Category)
                  AND NOT coalesce(nb.nodeId, '') IN $placeholder_ids
                RETURN
                    f.nodeId AS seed_id, f.name AS seed_name, labels(f) AS seed_labels,
                    properties(f) AS seed_props, COUNT {{ (f)--() }} AS seed_degree,
                    type(r) AS rel_type, properties(r) AS rel_props,
                    startNode(r).nodeId AS rel_start, startNode(r).name AS rel_start_name,
                    endNode(r).nodeId AS rel_end, endNode(r).name AS rel_end_name,
                    nb.nodeId AS nb_id, nb.name AS nb_name, labels(nb) AS nb_labels,
                    properties(nb) AS nb_props, COUNT {{ (nb)--() }} AS nb_degree
                ORDER BY nb_degree ASC
                LIMIT $row_limit
                """
                try:
                    remaining = max(max_nodes - len(current_ids), 0)
                    if remaining > 0:
                        hop2_rows = [dict(record) for record in session.run(hop2_cypher, {
                            "frontier_ids": frontier_ids,
                            "frontier_names": frontier_names,
                            "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                            "row_limit": max(remaining * 2, 20),
                        })]
                        # 阶段二只取新增的邻居与关系；返回的"中心节点"其实就是 frontier 自身，
                        # 它们已经作为一跳邻居存在于 connected_nodes 中，故此处有意忽略（下划线前缀）
                        _extra_central, extra_nodes, extra_rels = self._assemble_from_rows(
                            hop2_rows, remaining, exclude_ids=current_ids,
                        )
                        connected_nodes.extend(extra_nodes)
                        relationships.extend(extra_rels)
                        if extra_nodes:
                            depth_reached = 2
                except Exception as e:
                    logger.warning(f"子图阶段二查询失败（保持一跳结果）: {e}")

        # 【跨阶段关系去重】两个阶段各自的 seen_rel 是调用内局部集合，互相看不见，
        # 二跳回连到种子的边会被重复追加一次。此处按与写入时完全一致的签名口径统一收口，
        # 保证 relationship_count / density 与图结构自洽（详见 _dedupe_relationships）。
        relationships = self._dedupe_relationships(relationships)

        # ---------------- 图指标计算 ----------------
        node_count = len(central_nodes) + len(connected_nodes)
        rel_count = len(relationships)
        # 无向图密度：D = 2E / (N(N-1))，N<=1 时无意义，显式置 0
        density = (2.0 * rel_count) / (node_count * (node_count - 1)) if node_count > 1 else 0.0
        all_degrees = [int(n.get("degree") or 0) for n in central_nodes + connected_nodes]
        avg_degree = sum(all_degrees) / float(len(all_degrees)) if all_degrees else 0.0

        metrics: Dict[str, Any] = {
            "node_count": float(node_count),
            "relationship_count": float(rel_count),
            "density": round(float(min(density, 1.0)), 6),
            "avg_degree": round(avg_degree, 3),
            "max_degree": float(max(all_degrees) if all_degrees else 0),
            "depth_reached": float(depth_reached),
            "hub_pruned": float(hub_pruned),
        }

        return KnowledgeSubgraph(
            central_nodes=central_nodes,
            connected_nodes=connected_nodes,
            relationships=relationships,
            graph_metrics=metrics,
            reasoning_chains=[],
        )

    def _interleave_rows_by_seed(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把「(种子, 关系, 邻居)」行按种子轮转排序，用于多种子场景下的预算公平分配。

        规则：按 seed_id 首次出现的顺序把行分组（组内保持原有顺序，即 nb_degree DESC），
        然后依次从每组取一行轮转输出。单种子（或 rows 少于两个不同种子）时返回原列表，
        保证最常见的查询路径行为完全不变。

        Args:
            rows: 查询返回的扁平三元组（需含 seed_id 列）

        Returns:
            轮转后的新列表（不修改入参）
        """
        if not rows:
            return rows
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("seed_id") or "")
            groups.setdefault(key, []).append(row)
        if len(groups) <= 1:
            return rows
        interleaved: List[Dict[str, Any]] = []
        cursors = {key: 0 for key in groups}
        remaining = len(rows)
        while remaining > 0:
            for key, group in groups.items():
                cursor = cursors[key]
                if cursor < len(group):
                    interleaved.append(group[cursor])
                    cursors[key] = cursor + 1
                    remaining -= 1
        return interleaved

    def _assemble_from_rows(self, rows: List[Dict[str, Any]], max_nodes: int,
                            exclude_ids: Optional[Set[str]] = None) -> Tuple[
                                List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """把「(种子, 关系, 邻居)」三元组行组装成 中心节点 / 邻居节点 / 关系 三段结构。

        Args:
            rows: 查询返回的扁平三元组
            max_nodes: 邻居节点数量上限
            exclude_ids: 需要排除的 nodeId（用于阶段二避免重复纳入已有节点）

        Returns:
            (central_nodes, connected_nodes, relationships)；三段均已完成去重与脏数据过滤
        """
        exclude = exclude_ids or set()
        central_nodes: List[Dict[str, Any]] = []
        connected_nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        seen_central: Set[str] = set()
        seen_connected: Set[str] = set()
        seen_rel: Set[Tuple[str, str, str]] = set()

        for row in rows:
            seed_id = str(row.get("seed_id") or "")
            seed_node = {
                "id": seed_id,
                "node_id": seed_id,
                "name": str(row.get("seed_name") or ""),
                "labels": list(row.get("seed_labels") or []),
                "label": (list(row.get("seed_labels") or []) or [""])[0],
                "properties": dict(row.get("seed_props") or {}),
                "degree": int(row.get("seed_degree") or 0),
            }
            # 种子同样按"身份键"去重：无 nodeId 的节点（Category 等）靠 name 区分，
            # 避免多个不同 Category 都被当成同一个空串节点
            seed_key = self._node_identity_key(seed_node)
            if seed_key and seed_key not in seen_central and self._is_meaningful_node(seed_node):
                central_nodes.append(seed_node)
                seen_central.add(seed_key)

            nb_node = {
                "id": str(row.get("nb_id") or ""),
                "node_id": str(row.get("nb_id") or ""),
                "name": str(row.get("nb_name") or ""),
                "labels": list(row.get("nb_labels") or []),
                "label": (list(row.get("nb_labels") or []) or [""])[0],
                "properties": dict(row.get("nb_props") or {}),
                "degree": int(row.get("nb_degree") or 0),
            }
            # 【关键】这里必须用身份键而不是 nb_id 做真值判断与去重。
            # 原写法 `if not nb_id ... continue` 会把所有无 nodeId 的邻居（Category /
            # ConceptType / DifficultyLevel / RecipeCategory 全部无 nodeId）静默丢弃，
            # 是"子图里 Category 恒为 0"的根因 —— 上游 Cypher 的 NULL 安全过滤已经把
            # 荤菜 这类节点放行了，却在 Python 组装这一步又被判掉。
            nb_key = self._node_identity_key(nb_node)
            if not nb_key:
                # 既无 nodeId 也无 name 的脏节点，无法标识，直接丢弃
                continue
            if nb_key in seen_connected or nb_key in exclude:
                # 节点已收录，本轮不重复加入节点列表；但这一行代表的边仍是新边
                # （平行边 / 跨种子边 / 二跳回连边），必须记录，否则图结构会缺边。
                # 【此处不能再用 len(connected_nodes) < max_nodes 设闸】
                # 进入本分支的前提就是 nb 已在本轮子图内（在 seen_connected 或
                # 在 exclude 指定的已收录集合里），所以补进去的边【两端必然都在子图内】，
                # 不存在"悬空边"风险。此前把补边绑在节点预算上，导致预算一旦用尽，
                # 连子图内部真实存在的关系也被静默丢弃（该分支当时连日志都没有）：
                # 实测单实体「水煮鱼」depth=1 max_nodes=10 时，20 条两端均在子图内的边
                # 只记录了 19 条；多实体场景丢掉的则是独有证据
                #（如 ['鸡蛋','咖喱炒蟹'] 丢 REQUIRES 咖喱炒蟹→鸡蛋）。
                # 节点预算与关系记录是两件事，关系不受节点预算约束。
                self._append_relationship(relationships, seen_rel, row)
                continue
            if len(connected_nodes) >= max_nodes:
                continue
            if not self._is_meaningful_node(nb_node):
                continue
            connected_nodes.append(nb_node)
            seen_connected.add(nb_key)
            self._append_relationship(relationships, seen_rel, row)

        return central_nodes, connected_nodes, relationships

    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象。

        兼容两种输入：
          - 本模块内部使用的规范化载荷（含 central_nodes / connected_nodes / relationships / metrics 四个键）；
          - 早期版本的 neo4j Record（含 source / nodes / rels / metrics 键），保留以兼容既有调用。
        """
        try:
            if isinstance(record, dict) and "central_nodes" in record:
                return KnowledgeSubgraph(
                    central_nodes=list(record.get("central_nodes") or []),
                    connected_nodes=list(record.get("connected_nodes") or []),
                    relationships=list(record.get("relationships") or []),
                    graph_metrics=dict(record.get("metrics") or {}),
                    reasoning_chains=list(record.get("reasoning_chains") or []),
                )

            # 兼容路径：早期 Record 形态
            central_nodes = [self._node_to_dict(record["source"])] if record.get("source") is not None else []
            connected_nodes = [self._node_to_dict(node) for node in (record.get("nodes") or [])]
            relationships = [
                self._rel_to_dict(rel) if not isinstance(rel, list) else self._rel_to_dict(rel[0])
                for rel in (record.get("rels") or [])
                if rel is not None
            ]
            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=dict(record.get("metrics") or {}),
                reasoning_chains=[]
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )

    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取（真实可用的降级，而非空壳）。

        三级降级，逐级放宽：
          1. 图谱侧只做最简单的一次 1 跳邻居查询（不带 COUNT{}、不带聚合、不带路径模式，最大化成功率）；
          2. 失败则用预热的 entity_cache 在内存里构造一个最小子图（Neo4j 抖动时仍能给出实体级证据）；
          3. 全部失败才返回空子图，并在 metrics 里标注降级来源，便于上层观测与排障。
        """
        degraded_metrics: Dict[str, Any] = {"degraded": 1.0}

        # 降级路径的节点预算【必须沿用调用方传入的 max_nodes】。
        # 此前一级降级把上限硬编码为 80 并把 Cypher 的 row_limit 硬编码为 200，
        # 完全忽略 graph_query.max_nodes —— 而调用方（含 graph_rag_search 的透传）
        # 正是靠这个参数控制 prompt 长度。调用方把 max_nodes 压到 10 以适配上下文时，
        # 降级路径反而给出 80 个节点，降级结果比正常结果还重，与"降级应当更轻"的直觉相悖。
        fallback_max_nodes = _clamp(getattr(graph_query, "max_nodes", 50),
                                   _MIN_SUBGRAPH_NODES, _MAX_SUBGRAPH_NODES, 50)

        # 一级降级：最简 Cypher
        try:
            with self.driver.session() as session:
                seeds = self._resolve_seed_nodes(graph_query.source_entities, limit=3, session=session)
                if seeds:
                    rows = [dict(r) for r in session.run(
                        """
                        MATCH (seed) WHERE seed.nodeId IN $seed_ids
                        MATCH (seed)-[r]-(nb)
                        WHERE (nb:Recipe OR nb:Ingredient OR nb:CookingStep OR nb:Category)
                          AND NOT coalesce(nb.nodeId, '') IN $placeholder_ids
                        RETURN seed.nodeId AS seed_id, seed.name AS seed_name, labels(seed) AS seed_labels,
                               properties(seed) AS seed_props, 0 AS seed_degree,
                               type(r) AS rel_type, properties(r) AS rel_props,
                               startNode(r).nodeId AS rel_start, startNode(r).name AS rel_start_name,
                               endNode(r).nodeId AS rel_end, endNode(r).name AS rel_end_name,
                               nb.nodeId AS nb_id, nb.name AS nb_name, labels(nb) AS nb_labels,
                               properties(nb) AS nb_props, 0 AS nb_degree
                        LIMIT $row_limit
                        """,
                        {
                            "seed_ids": [s["id"] for s in seeds],
                            "placeholder_ids": list(_PLACEHOLDER_NODE_IDS),
                            "row_limit": max(fallback_max_nodes * 4, 50),
                        },
                    )]
                    central_nodes, connected_nodes, relationships = self._assemble_from_rows(
                        rows, fallback_max_nodes)
                    # 与主路径同口径的跨列表收口：种子若同时是另一个种子的邻居，
                    # 不能既算中心节点又算邻居，否则 node_count 重复计数
                    _central_keys = {self._node_identity_key(n) for n in central_nodes}
                    _central_keys.discard("")
                    if _central_keys:
                        connected_nodes = [n for n in connected_nodes
                                           if self._node_identity_key(n) not in _central_keys]
                    if central_nodes or connected_nodes:
                        node_count = len(central_nodes) + len(connected_nodes)
                        rel_count = len(relationships)
                        density = (2.0 * rel_count) / (node_count * (node_count - 1)) if node_count > 1 else 0.0
                        self._attach_degrees(central_nodes + connected_nodes)
                        # avg_degree / max_degree 与主路径保持同一套字段：
                        # 描述文本 (_build_subgraph_description) 直接读 metrics['avg_degree']，
                        # 一级降级不写这两个字段时它会走 `metrics.get('avg_degree', 0)` 兜底，
                        # 于是在度数明明已经补好的情况下，喂给生成模型的文本写着「平均度数 0」——
                        # 降级路径本来就最需要自证可信，这种"数据齐全却报 0"的表述反而最伤可信度。
                        _degrees = [int(n.get("degree") or 0) for n in (central_nodes + connected_nodes)]
                        metrics = {
                            "node_count": float(node_count),
                            "relationship_count": float(rel_count),
                            "density": round(float(min(density, 1.0)), 6),
                            "avg_degree": round(sum(_degrees) / float(len(_degrees)), 3) if _degrees else 0.0,
                            "max_degree": float(max(_degrees)) if _degrees else 0.0,
                            "depth_reached": 1.0,
                            "source": 1.0,
                        }
                        metrics.update(degraded_metrics)
                        logger.info(f"子图一级降级成功：{node_count} 节点 / {rel_count} 关系")
                        return KnowledgeSubgraph(
                            central_nodes=central_nodes,
                            connected_nodes=connected_nodes,
                            relationships=relationships,
                            graph_metrics=metrics,
                            reasoning_chains=[],
                        )
        except Exception as e:
            logger.warning(f"子图一级降级失败: {e}")

        # 二级降级：用内存索引构造最小子图
        try:
            names = [self._normalize_entity_name(e) for e in (graph_query.source_entities or [])]
            names = [n for n in names if n]
            cache_seeds = self._resolve_seed_nodes_from_cache([n.lower() for n in names], limit=3) if names else []
            if cache_seeds:
                # 把内存索引里与种子同类/同名的节点也纳入，形成最小可用知识网络
                # 节点预算同样沿用调用方传入的 max_nodes（此前硬编码 20，会无视调用方
                # 为控制 prompt 长度而下调的上限），同时不超过 20 以免降级结果过重
                _level2_budget = min(20, fallback_max_nodes)
                _seed_ids = {str(s.get("id") or "") for s in cache_seeds}
                connected: List[Dict[str, Any]] = []
                for node_id, info in self.entity_cache.items():
                    if len(connected) >= _level2_budget:
                        break
                    node_name = str(info.get("name") or "")
                    if not node_name or node_name in _PLACEHOLDER_NAMES:
                        continue
                    # 种子自身也会命中下面的子串条件（s["name"] in node_name 对 s["name"]==node_name
                    # 恒为真），若不排除就会既算中心节点又算关联节点，node_count 重复计数
                    if str(node_id) in _seed_ids:
                        continue
                    if any(s["name"] and s["name"] in node_name for s in cache_seeds):
                        connected.append({
                            "id": str(node_id),
                            "node_id": str(node_id),
                            "name": node_name,
                            "labels": list(info.get("labels") or []),
                            "label": (list(info.get("labels") or []) or [""])[0],
                            "properties": {"name": node_name, "category": info.get("category")},
                            "degree": int(info.get("degree") or 0),
                        })
                metrics = {
                    "node_count": float(len(cache_seeds) + len(connected)),
                    "relationship_count": 0.0,
                    "density": 0.0,
                    "depth_reached": 0.0,
                    "source": 2.0,
                }
                metrics.update(degraded_metrics)
                logger.info(f"子图二级降级（内存索引）：{len(cache_seeds)} 中心 / {len(connected)} 关联")
                return KnowledgeSubgraph(
                    central_nodes=cache_seeds,
                    connected_nodes=connected,
                    relationships=[],
                    graph_metrics=metrics,
                    reasoning_chains=[],
                )
        except Exception as e:
            logger.warning(f"子图二级降级失败: {e}")

        # 三级降级：确实拿不到任何证据
        logger.error("子图降级全部失败，返回空子图")
        metrics = {"node_count": 0.0, "relationship_count": 0.0, "density": 0.0, "source": 0.0}
        metrics.update(degraded_metrics)
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics=metrics,
            reasoning_chains=[]
        )

    # -------------------------------------------------------------------------
    # 图结构推理
    # -------------------------------------------------------------------------

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于图结构的推理：这是图RAG的智能之处
        不仅检索信息，还能进行逻辑推理
        """
        reasoning_chains = []

        try:
            # 1. 识别推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)

            # 2. 构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)

            # 3. 验证推理链的可信度
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)

            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains

        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []

    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别子图里【真实存在】的推理模式。

        与早期「返回写死的三个模式」不同，这里完全由子图实际包含的关系类型与节点结构决定：
        某个模式只有在它的证据关系确实出现时才会被列出，因此不同子图会得到不同的模式集合，
        后续 _build_reasoning_chain 才有真实拓扑可依据。
        """
        if subgraph is None:
            return []

        rel_types: Set[str] = {str(r.get("type") or "") for r in (subgraph.relationships or [])}
        node_labels: Set[str] = set()
        for node in list(subgraph.central_nodes or []) + list(subgraph.connected_nodes or []):
            node_labels.update(node.get("labels") or [])

        patterns: List[str] = []

        # 组成关系：菜谱 → 食材，是烹饪知识最基本的推理依据
        if "REQUIRES" in rel_types or ("Recipe" in node_labels and "Ingredient" in node_labels):
            patterns.append("组成关系")
        # 时序关系：步骤之间先后顺序
        if "NEXT_STEP" in rel_types or "CONTAINS_STEP" in rel_types or "CookingStep" in node_labels:
            patterns.append("时序关系")
        # 分类归属：菜谱 → 分类，支撑「同类菜品」推理
        if "BELONGS_TO_CATEGORY" in rel_types or "Category" in node_labels:
            patterns.append("分类归属")
        # 相似关系：显式 SIMILAR 边
        if "SIMILAR" in rel_types:
            patterns.append("相似关系")
        # 技法共性：同工具 / 同技法，跨菜谱的精细连通依据
        if "USES_SAME_METHOD" in rel_types or "USES_SAME_TOOL" in rel_types:
            patterns.append("技法共性")
        # 食材替代：同一子图内存在同类食材（如多个「调料」）才成立
        if self._has_sibling_ingredients(subgraph):
            patterns.append("食材替代")
        # 难度分层
        if "DIFFICULTY_LEVEL" in rel_types or "DifficultyLevel" in node_labels:
            patterns.append("难度分层")
        # 概念归类
        if "HAS_CONCEPT_TYPE" in rel_types or "ConceptType" in node_labels:
            patterns.append("概念归类")

        return patterns

    def _has_sibling_ingredients(self, subgraph: KnowledgeSubgraph) -> bool:
        """判断子图内是否存在「同类食材」（用于素材替代推理）。

        烹饪场景下最有价值的推理之一就是「缺某样调料时用什么替代」，
        它的图结构依据就是两个食材节点挂到同一个 Category 上。
        """
        if subgraph is None:
            return False
        category_to_ingredients: Dict[str, Set[str]] = defaultdict(set)
        for node in (subgraph.connected_nodes or []):
            labels = set(node.get("labels") or [])
            if "Ingredient" not in labels:
                continue
            category = str((node.get("properties") or {}).get("category") or "")
            if category:
                category_to_ingredients[category].add(str(node.get("name") or ""))
        return any(len(items) >= 2 for items in category_to_ingredients.values())

    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """按推理模式，用子图里的真实节点与边构造一条可读、可溯源的推理链。

        设计原则：每条链都必须【引用具体实体名】并【说明依据的关系类型】，
        这样它才能作为证据喂给生成模型；返回 None 表示该模式在当前子图上证据不足，不应产出噪声推理。
        """
        if subgraph is None:
            return None

        central_names = [str(n.get("name") or "") for n in (subgraph.central_nodes or []) if n.get("name")]
        central_text = "、".join(f"「{n}」" for n in central_names[:3]) if central_names else "该实体"

        # 按标签与关系类型把节点/边分好类，供各模式取用
        ingredients = [n for n in (subgraph.connected_nodes or []) if "Ingredient" in (n.get("labels") or [])]
        steps = [n for n in (subgraph.connected_nodes or []) if "CookingStep" in (n.get("labels") or [])]
        categories = [n for n in (subgraph.connected_nodes or []) if "Category" in (n.get("labels") or [])]
        recipes = [n for n in (subgraph.connected_nodes or []) if "Recipe" in (n.get("labels") or [])]
        difficulties = [n for n in (subgraph.connected_nodes or []) if "DifficultyLevel" in (n.get("labels") or [])]

        try:
            if pattern == "组成关系":
                if not ingredients:
                    return None
                by_category: Dict[str, List[str]] = defaultdict(list)
                for item in ingredients:
                    category = str((item.get("properties") or {}).get("category") or "未分类")
                    by_category[category].append(str(item.get("name") or ""))
                groups = []
                for category, names in sorted(by_category.items(), key=lambda kv: -len(kv[1]))[:3]:
                    groups.append(f"{category}类（{'、'.join([n for n in names if n][:4])}）")
                return (f"{central_text}的构成关系：共关联 {len(ingredients)} 种食材，"
                        f"按类别可分为 {'；'.join(groups)}。依据关系类型 REQUIRES。")

            if pattern == "时序关系":
                if not steps:
                    return None

                # 必须先按「所属菜谱」把步骤分组，否则会把多道菜的步骤混排成一条假工序。
                # 实测教训：一次子图里可能同时含 5 道川菜的步骤，它们的步骤名都叫「步骤1/步骤2…」，
                # 若不分组就会输出「步骤1 → 步骤2 → 步骤2 → 步骤2 → 步骤3」这种看似工序、实为串台的序列。
                owner_of_step: Dict[str, str] = {}
                for rel in (subgraph.relationships or []):
                    if str(rel.get("type") or "") in ("CONTAINS_STEP", "NEXT_STEP"):
                        start_id, end_id = str(rel.get("start_id") or ""), str(rel.get("end_id") or "")
                        if start_id and end_id and end_id not in owner_of_step:
                            owner_of_step[end_id] = start_id

                grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for step in steps:
                    grouped[owner_of_step.get(str(step.get("id") or ""), "__unowned__")].append(step)

                def _step_order(step: Dict[str, Any]) -> float:
                    """按 stepNumber 排序；缺失/非数值的排在最后。"""
                    value = (step.get("properties") or {}).get("stepNumber")
                    return value if isinstance(value, (int, float)) else 999.0

                # 优先展示有明确归属菜谱的组；无归属的步骤组放最后，并如实说明它无法确定工序
                sequences: List[str] = []
                for owner_id, group in sorted(
                    grouped.items(), key=lambda kv: (kv[0] == "__unowned__", -len(kv[1]))
                ):
                    if len(sequences) >= 3:
                        break
                    ordered = sorted(group, key=_step_order)
                    names = [str(s.get("name") or "") for s in ordered[:8]]
                    seq = " → ".join([n for n in names if n])
                    if not seq:
                        continue
                    if owner_id == "__unowned__":
                        sequences.append(f"{len(group)} 个步骤（未关联到具体菜谱，无法确定先后工序）：{seq}")
                    else:
                        owner_name = next(
                            (str(n.get("name") or "") for n in (subgraph.central_nodes or [])
                             if str(n.get("id") or "") == owner_id),
                            "",
                        )
                        prefix = f"「{owner_name}」" if owner_name else "某道菜"
                        sequences.append(f"{prefix}共 {len(group)} 步，依序为：{seq}")

                if not sequences:
                    return None
                return ("制作流程：" + "；".join(sequences) + "。依据关系类型 CONTAINS_STEP / NEXT_STEP。")

            if pattern == "分类归属":
                if not categories:
                    return None
                names = [str(c.get("name") or "") for c in categories][:5]
                return (f"{central_text}在分类体系中归属于 {'、'.join([n for n in names if n])}，"
                        f"因此可与同分类下的其他条目互为参照。依据关系类型 BELONGS_TO_CATEGORY。")

            if pattern == "相似关系":
                if not recipes:
                    return None
                names = [str(r.get("name") or "") for r in recipes][:5]
                return (f"{central_text}与 {'、'.join([n for n in names if n])} 之间存在显式相似关系，"
                        f"可作为同类替代或对比的候选。依据关系类型 SIMILAR。")

            if pattern == "技法共性":
                methods = _dedup_preserve_order([
                    str((r.get("properties") or {}).get("method") or (r.get("properties") or {}).get("tool") or "")
                    for r in (subgraph.relationships or [])
                    if str(r.get("type") or "") in ("USES_SAME_METHOD", "USES_SAME_TOOL")
                ])
                methods = [m for m in methods if m]
                if not methods:
                    return None
                return (f"{central_text}在制作技法上与其它菜谱存在共性，共享的技法/工具包括："
                        f"{'、'.join(methods[:5])}。依据关系类型 USES_SAME_METHOD / USES_SAME_TOOL。")

            if pattern == "食材替代":
                by_category: Dict[str, List[str]] = defaultdict(list)
                for item in ingredients:
                    category = str((item.get("properties") or {}).get("category") or "")
                    name = str(item.get("name") or "")
                    if category and name:
                        by_category[category].append(name)
                substitutes = [
                    (category, names) for category, names in by_category.items() if len(names) >= 2
                ]
                if not substitutes:
                    return None
                parts = [
                    f"缺少{category}类食材时，可在 {'、'.join(names[:4])} 之间择一替代"
                    for category, names in sorted(substitutes, key=lambda kv: -len(kv[1]))[:3]
                ]
                return (f"{central_text}的食材替代参考：" + "；".join(parts) + "。依据同一 Category 下食材的可互换性。")

            if pattern == "难度分层":
                if not difficulties:
                    return None
                names = [str(d.get("name") or "") for d in difficulties][:5]
                return (f"{central_text}处于 {'、'.join([n for n in names if n])} 难度层级，"
                        f"可用于按难度筛选或排序同类菜谱。依据关系类型 DIFFICULTY_LEVEL。")

            if pattern == "概念归类":
                concepts = [str(n.get("name") or "") for n in (subgraph.connected_nodes or [])
                            if "ConceptType" in (n.get("labels") or [])]
                concepts = [c for c in concepts if c]
                if not concepts:
                    return None
                return (f"{central_text}在概念层被归类为 {'、'.join(concepts[:3])}。"
                        f"注意该关系覆盖范围极广，推理时区分度有限。依据关系类型 HAS_CONCEPT_TYPE。")

            return None
        except Exception as e:
            logger.warning(f"构造推理链失败（模式={pattern}）: {e}")
            return None

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """对推理链做可信度校验与排序（替代早期「只截前三条」的假校验）。

        校验口径（只依赖文本自身与查询，无需隐藏状态）：
          - 与查询的词法重合度：链条若完全答非所问则应当降权；
          - 具体性：是否引用了具名实体（「…」）、是否带数量或步骤号 —— 越具体越可能是真推理而非套话；
        然后去重、按综合分降序、截断到固定条数。
        """
        if not chains:
            return []

        query_bigrams = _char_bigrams(query or "")
        scored: List[Tuple[float, str]] = []
        seen: Set[str] = set()

        for chain in chains:
            text = (chain or "").strip()
            if not text:
                continue
            signature = re.sub(r"\s+", "", text)
            if signature in seen:
                continue
            seen.add(signature)

            overlap = _containment(query_bigrams, _char_bigrams(text)) if query_bigrams else 0.0

            concreteness = 0.0
            if "「" in text and "」" in text:
                concreteness += 0.5          # 引用了具名实体
            if re.search(r"\d", text):
                concreteness += 0.3          # 含数量/序号等可核验细节
            if len(text) >= 20:
                concreteness += 0.2          # 足够长才承载得了推理

            score = 0.6 * overlap + 0.4 * concreteness
            scored.append((score, text))

        scored.sort(key=lambda item: -item[0])
        # 最多返回 5 条：推理链是给生成模型的辅助证据，过多反而稀释上下文
        return [text for _, text in scored[:5]]

    # -------------------------------------------------------------------------
    # 自适应查询规划
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # 主检索接口
    # -------------------------------------------------------------------------

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")

        # 【入口防御一：query 必须是字符串】
        # understand_graph_query 内部会 query.strip()，而它是在下面的 try 之前调用的，
        # int / list / dict / float 会抛 AttributeError 直接穿透公开入口（实测均复现）。
        # 检索模块对上层应当是"降级不抛错"的语义，这里统一收口为字符串。
        query = _coerce_query_text(query)

        # 【入口防御二：top_k 必须钳制成整数】
        # 此前零校验，直接 results[:top_k]：
        #   - top_k=None  → 切片键退化为 None，完全不截断，返回底层全部结果（实测 60 条）；
        #   - top_k 为负  → 按尾部截断，返回近全量（-1 → 59 条）；
        #   - 字符串/浮点 → 切片抛 TypeError，被下面的兜底 except 吞成 return []，
        #     把"参数类型错误"伪装成"图里没数据"，调用方无法区分。
        # 非整数输入记一条 warning 后回退到默认值，越界值钳到 [1, ...) ——
        # top_k<=0 对检索 API 没有意义，返回 0 条只会让调用方以为"图里没有证据"。
        try:
            top_k = int(top_k)
        except (TypeError, ValueError, OverflowError):
            logger.warning(f"top_k 非法（{top_k!r}），回退为默认值 5")
            top_k = 5
        if top_k < 1:
            logger.warning(f"top_k={top_k} 无意义（<=0），钳制为 1")
            top_k = 1

        if not self._is_driver_ready():
            logger.warning("Neo4j连接未建立，返回空结果")
            return []

        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")

        results = []

        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.SUBGRAPH:
                # 子图提取
                subgraph = self.extract_knowledge_subgraph(graph_query)

                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)

                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))

            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.CLUSTERING:
                # 聚类相似查询：走相似实体检索，产出"以相似项为成员"的路径型证据
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            # 2.5 结果回收：若主策略没拿到证据，退一步用子图策略补一次，
            #     避免「路径分支恰好为空」时对用户静默返回空结果
            if not results:
                logger.info("主策略未取得结果，回退到子图检索")
                subgraph_query = GraphQuery(
                    query_type=QueryType.SUBGRAPH,
                    source_entities=graph_query.source_entities,
                    target_entities=graph_query.target_entities,
                    max_depth=min(graph_query.max_depth, _MAX_TRAVERSAL_DEPTH),
                    max_nodes=graph_query.max_nodes,
                )
                subgraph = self.extract_knowledge_subgraph(subgraph_query)
                if subgraph.central_nodes or subgraph.connected_nodes:
                    reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                    results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))

            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)

            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]

        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []

    # ========== 辅助方法 ==========

    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录。

        支持两种形态：
          - 含 "path" 键的 neo4j Path 对象（推荐形态，能拿到真实方向与关系类型）；
          - 早期版本的 path_nodes / rels / path_len / relevance 扁平键（保留兼容）。
        关系类型一律通过 _rel_to_dict 读取（内部使用驱动的 rel.type），
        避免早期 `type(rel).__name__` 只得到 'Relationship' 的缺陷。
        """
        try:
            if record.get("path") is not None:
                neo_path = record["path"]
                nodes = [self._node_to_dict(n) for n in neo_path.nodes]
                rels = [self._rel_to_dict(r) for r in neo_path.relationships]
                length = len(rels)
                relevance = record.get("relevance")
                # 查询里带了 relevance（例如全文/向量侧已经打过分的图路径）就直接沿用，
                # 否则交回 _score_path 统一评分，保证同一批结果的分数量纲一致
                if isinstance(relevance, (int, float)):
                    return GraphPath(
                        nodes=nodes,
                        relationships=rels,
                        path_length=length,
                        relevance_score=float(relevance),
                        path_type="multi_hop",
                    )
                built = self._build_graph_path(
                    nodes=nodes, rels=rels, path_type="multi_hop",
                    path_length=length if length else None,
                )
                if built is not None:
                    built.relevance_score = self._score_path(built)
                return built

            # 兼容路径：扁平键形态
            path_nodes = []
            for node in (record.get("path_nodes") or []):
                if isinstance(node, dict):
                    path_nodes.append(node)
                else:
                    path_nodes.append(self._node_to_dict(node))

            relationships = []
            for rel in (record.get("rels") or []):
                if isinstance(rel, dict):
                    relationships.append(rel)
                elif isinstance(rel, list):
                    relationships.extend([self._rel_to_dict(item) for item in rel])
                else:
                    relationships.append(self._rel_to_dict(rel))

            relevance = record.get("relevance")
            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=int(record.get("path_len") or max(len(path_nodes) - 1, 1)),
                relevance_score=float(relevance) if isinstance(relevance, (int, float)) else 0.0,
                path_type="multi_hop"
            )

        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None

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
