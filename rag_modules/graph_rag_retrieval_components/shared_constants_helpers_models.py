"""图 RAG 共享底座：图结构常量、模块级纯函数、查询/路径/子图数据结构。"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

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
