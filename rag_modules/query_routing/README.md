# 查询路由维护地图

这份文档只回答三个问题：请求怎么走、常见改动改哪里、出错怎么排查。

包的位置是 `rag_modules/query_routing/`，包外入口是
`IntelligentQueryRouter.route_query(...)`。路由包本身不生成最终回答，只负责
“读懂问题 + 取到证据”，回答交给 `generation_integration.py`。

## 1. 它和系统其它部分的关系

```text
POST /api/chat
  → WebServiceHandler._chat_once() / _stream_events()
  → IntelligentQueryRouter.route_query(query, top_k=5, session_id, session_state)
  → (List[Document], QueryAnalysis)
  → GenerationIntegrationModule.generate_adaptive_answer(...)

命令行入口：
AdvancedGraphRAGSystem.ask_question_with_routing()
  → IntelligentQueryRouter.route_query(...)
```

对外签名保持不变：

```python
route_query(query, top_k=5, session_id="", session_state=None)
    -> (List[Document], QueryAnalysis)
```

新增能力优先加到 `QueryAnalysis` 的字段里，不要改签名和返回顺序。

`QueryAnalysis.response_type` 有四种取值：

| 值 | 含义 | 是否检索 |
| --- | --- | --- |
| `answer` | 正常回答 | 是 |
| `clarify` | 信息不足，先澄清 | 否 |
| `refuse` | 域外问题，礼貌拒绝 | 否 |
| `no_data` | 检索后没有可用证据 | 是（但结果为空或全部 missing） |

`QueryAnalysis.evidence_status` 取
`complete / partial / missing / not_required / fallback`，只描述证据情况，
不决定最终话术。

## 2. 一次请求的完整链路

```text
route_query
  0. 读会话状态
     active_recipe / active_topic / constraints / recent_user_queries
  1. understanding.understand()
     LLM 结构化输出 → JSON 解析 → 失败修复一次 → 仍失败则规则兜底
     同时完成实体/槽位规范化、指代补全、澄清判断
  2. policy.plan()
     纯规则：多菜拆分、多意图拆分、置信度门控、子查询、top_k
  3. evidence.execute()
     每个 SubPlan 按 engine 取 graph / text / tips 证据
  4. validators.validate()
     每路子问题给出 complete / partial / missing
  5. 合并去重
     返回文档和 QueryAnalysis
  6. generation_integration
     单意图直接生成；多意图按子问题分节生成
```

`rules.py` 不是独立的一层，它被理解层调用：LLM 正常时只做措辞修正，LLM
失败时整条规则链路兜底。策略层不再调用 LLM。

关键原则：

- LLM 只负责语义理解（`route_key`、`span`、`entities`、`slots`、`confidence`）；
- `RouteKeySpec` 是路由元数据的唯一来源（证据形态、来源、引擎、校验器）；
- 路由业务词表只放在 `domain.py`，`rules.py` / `slots.py` 只做匹配和清洗；
- 跨轮保留哪些槽位由 `DomainConfig.persisted_slots` 决定；
- 理解失败收敛到规则，执行异常降级到传统检索，日志落盘失败只告警；
- 多意图检索并行、输出按序，不能因为并发改变回答顺序。

### 失败时的兜底顺序

| 环节 | 失败表现 | 兜底 |
| --- | --- | --- |
| LLM 理解 | 超时、JSON 非法、字段缺失 | 修复一次，仍失败走 `rule_understand` |
| 策略规划 | 没有有效意图 | 返回 `clarify`，不检索 |
| 证据执行 | 单个子计划抛异常 | 该子计划为空并标记 `missing` |
| 整个路由 | 计划执行整体异常 | 传统 hybrid 检索，`evidence_status="fallback"` |
| 决策日志 | 目录不可写、磁盘满 | 告警一次，不影响回答 |

## 3. 核心数据结构

通用结构在 `schemas.py`，对外结果 `QueryAnalysis` 在
`intelligent_query_router.py`。

| 结构 | 作用 |
| --- | --- |
| `EntityRef` | 查询里的一个实体：菜名、食材、菜系、分类 |
| `IntentItem` | 一个可独立路由的意图；多意图时就是列表多项 |
| `ContextInfo` | 多轮上下文：是否用历史、是否换话题、当前菜 |
| `Understanding` | 理解层输出，也是理解层与策略层之间的契约 |
| `SubPlan` | 一个子问题的执行计划：查询、引擎、top_k、校验器 |
| `RoutePlan` | 策略层输出：是否检索、子计划、回答类型 |
| `ValidationResult` | 单个子计划的证据覆盖结果 |
| `QueryAnalysis` | 路由对外结果，兼容旧调用方，也用于日志和解释 |

`Understanding.source` 的取值：

- `llm`：模型首次输出解析成功；
- `llm_repair`：模型输出有问题，修复 Prompt 后成功；
- `rule`：模型失败，规则兜底；
- `legacy`：旧接口字段转换而来。

`SubPlan.engine` 的取值：

```text
none / graph_entity / graph_relation / hybrid / combined
tips_pairing / tips_safety / tips_technique / tips_rule
```

`SubPlan.depends_on` 只表示生成顺序，不表示检索依赖。检索仍然并行执行。

## 4. 六个顶层意图 → 十二个路由键

顶层意图只用于粗分类和 Prompt 展示，真正决定证据怎么取的是路由键。

| 路由键 | 顶层意图 | engine | validator | 证据来源 |
| --- | --- | --- | --- | --- |
| `recipe_ingredients` | `recipe_detail` | `graph_entity` | `list` | `REQUIRES` 全量 + 菜谱原文 |
| `recipe_steps_time` | `recipe_detail` | `graph_entity` | `steps` | `CONTAINS_STEP` + 原文操作 + tags |
| `recipe_overview` | `recipe_detail` | `graph_entity` | `overview` | Recipe 属性 + 菜谱原文 |
| `ingredient_pairing` | `ingredient_info` | `tips_pairing` | `pair` | tips：食材相克与禁忌 |
| `ingredient_property` | `ingredient_info` | `tips_technique` | `technique` | tips：learn/*.md + 原文 |
| `recommend_list` | `recommendation` | `graph_entity` | `recommend` | Recipe 的 tags/name/description |
| `recommend_scenario` | `recommendation` | `tips_rule` | `scenario` | tips：如何选择现在吃什么 + 图候选 |
| `nutrition_value` | `nutrition_health` | `graph_entity` | `value` | 菜谱原文精确数值 |
| `nutrition_benefit` | `nutrition_health` | `graph_entity` | `benefit` | 菜谱原文 + 向量兜底 |
| `food_safety` | `nutrition_health` | `tips_safety` | `safety` | tips：learn/食品安全.md |
| `chitchat` | `chitchat` | `none` | — | 不检索 |
| `out_of_scope` | `out_of_scope` | `none` | — | 不检索 |

`recipe_steps_time` 和 `nutrition_value` 使用 `value_policy="exact_only"`，
没有精确证据时不能估算。`ingredient_pairing`、`nutrition_benefit`、
`food_safety` 会带免责声明。`out_of_scope` 的 `refuse=True`。

### RouteKeySpec 的字段

必填字段：

```text
key / intent / evidence_profile / engine / source
need_complete / default_top_k / validator / keywords
```

常用可选开关：

| 字段 | 作用 |
| --- | --- |
| `entity_first` | 子查询里优先靠实体绑定菜谱 |
| `expand_per_recipe` | 一句话点了多道菜时，每道菜拆一个子计划 |
| `prefix_shared_recipe` | 多意图省略主语的子问题自动补共享菜名 |
| `depends_on_previous` | 生成时需要排在指定路由键之后 |
| `boost_with_entity` | 规则兜底时命中实体，置信度抬到 0.8 |
| `topic_mode` | 会话主题键的特殊处理（`pair` / `chitchat`） |
| `disclaimer` | 回答末尾加一般性饮食建议 |
| `refuse` | 该路由直接拒绝，不回答 |
| `reasoning_required` | 标记需要解释/推理的回答 |
| `value_policy` | 数值策略，如 `exact_only` |

`IntentItem.engine_override` 只为兼容旧版 LLM 输出保留，正常路由以
`RouteKeySpec.engine` 为准。

## 5. 置信度门控

默认阈值在 `domain.RouteThreshold`：

| 阈值 | 默认值 | 作用 |
| --- | --- | --- |
| `min_execute` | `0.50` | 低于它先澄清，不猜 |
| `retrieval_override` | `0.70` | 达到它覆盖 LLM 的“不检索” |
| `no_retrieval` | `0.80` | 闲聊/域外意图达到它可以拦截检索 |

阈值可以按路由键单独覆盖，也可以整体外置：

```bash
ROUTE_THRESHOLDS_PATH=/path/to/thresholds.json python main.py
```

```json
{
  "default": {"min_execute": 0.5, "retrieval_override": 0.7, "no_retrieval": 0.8},
  "recommend_list": {"min_execute": 0.6, "retrieval_override": 0.8, "no_retrieval": 0.8}
}
```

文件不存在或格式错误时使用默认值，不会阻断启动。

## 6. 文件职责

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 包内公开类型和服务导出 |
| `domain.py` | 意图、路由键、规则、词表、候选过滤、图谱/向量库 Schema |
| `schemas.py` | `Understanding / SubPlan / RoutePlan` 等通用结构 |
| `prompts.py` | 理解 Prompt、修复 Prompt、回答约束 `ANSWER_RULES` |
| `understanding.py` | LLM 调用/解析、实体规范化、上下文、澄清 |
| `rules.py` | 追问分类、规则路由、意图合并、纯约束声明 |
| `slots.py` | 槽位归一化和原文抽取 |
| `policy.py` | 置信度门控、多意图拆分、子查询、执行计划 |
| `evidence.py` | Neo4j / 菜谱原文 / tips / hybrid 证据 |
| `validators.py` | 证据覆盖校验 |
| `telemetry.py` | 路由决策 JSONL |
| `tips_knowledge.py` | CookWise tips 文档解析 |
| `text_utils.py` | 归一化、去重、实体词匹配等纯文本工具 |

## 7. 多轮状态契约

会话状态由 `session_cache_manager.py` 维护，路由包只读不写。

| 字段 | 含义 |
| --- | --- |
| `active_recipe` | 当前讨论的菜；追问“那要多少盐”靠它绑定 |
| `active_topic` | 当前话题，用于缓存和主题切换 |
| `constraints` | 跨轮约束，只保留 `persisted_slots` 里的槽位 |
| `last_route_key` / `last_intent` | 上一轮路由结果 |
| `recent_user_queries` | 最近用户原话，最多 4 条 |
| `updated_at` | 最后更新时间 |

当前跨轮保留的槽位：

```python
DOMAIN.persisted_slots = ("diet", "persona", "servings", "spicy", "exclude", "cuisine")
```

几条容易踩的规则：

- 理解层只拿最近 2 条用户原话 + 已知约束做指代消解，不拼接 AI 回答；
- 域外问题会清空 `active_recipe`，避免下一轮“它…”复用旧菜谱；
- 换话题时 `active_recipe` 换成新菜名；新话题没点菜就清空；
- Web 入口在每轮结束后调用 `update_state_after_turn`，但不会把 AI 答案写进去。

## 8. 多意图：并行检索，按序输出

策略层负责拆分：

- 一个意图里点了多道菜，且路由键 `expand_per_recipe=True`，每道菜拆一个子计划；
- 多个意图按顺序生成 `SubPlan`；
- 省略主语的子问题补共享菜名；推荐类通过 `prefix_shared_recipe=False` 关闭；
- `depends_on` 只影响生成顺序，不影响检索并发。

执行层：

- 单子计划直接执行，不启动事件循环；
- 多子计划用 `asyncio.to_thread` + `Semaphore(4)` 并行，最多同时 4 路；
- `asyncio.gather` 保持返回顺序，合并时按子计划顺序写回；
- 文档按 `(node_id, chunk_id, 正文前 80 字)` 去重；
- 每篇文档带 `sub_query_index / sub_query / sub_route_key`；
- 整体状态：全部有证据为 `complete`，任一 `missing/partial` 为 `partial`，
  全部为空为 `missing`。

生成层也是最多 4 路并行。流式模式下第 1 节边生成边输出，其余节并行生成、
按序吐字；如果第 1 节是“怎么做”且证据里有 3 步以上，首节也走非流式，先修复
步骤结构再输出。只要任一子问题有 `depends_on`，就不分节，改为单次生成并在
Prompt 里声明“先答被依赖的问题”。节标题使用 `### 1. 子问题`，不要改回
`1. …` 列表项，否则正文有序列表会接着标题继续编号。

## 9. 证据校验与降级

`validators.py` 的注册表：

```text
list / steps / overview / pair / technique
recommend / scenario / value / benefit / safety
```

校验只做确定性检查，不调用 LLM。结果有三态：

| 状态 | 含义 |
| --- | --- |
| `complete` | 该子问题的必需证据齐全 |
| `partial` | 有一部分证据，但覆盖不完整 |
| `missing` | 没有可用证据 |

路由层的补充规则：

- 单个子计划校验不完整且 `retry_hint="text"` 时，补一次文本检索；
- 用过历史改写、且整体证据不完整时，用原始 query 再做一次文本检索；
- 所有子问题都 `missing` 时，`answer` 改成 `no_data`，不能拿无关文本硬答；
- 整个计划执行异常时降级到传统 hybrid，`evidence_status="fallback"`。

## 10. 常见改动

### 加一个同义词

1. 优先加到 `domain.py` 的核心词常量（如 `FOOD_SAFETY_TERMS`），
   `ROUTE_SPECS.keywords` 和 `RULE_SPECS` 会一起生效。
2. 如果只影响理解 Prompt，改对应 `ROUTE_SPECS` 的 `keywords`。
3. 口语到标准词的映射加到 `domain.ALIASES`；槽位词加到 `SlotVocabulary`。
4. 跑 `python -m pytest -q tests/test_query_routing.py`。

### 加一个推荐过滤维度

1. 在 `domain.CANDIDATE_FILTERS` 加一条 `CandidateFilterSpec`。
2. `fields` 可选 `name / tags / category / cuisine / ingredient`。
3. `evidence.py` 按配置生成 Cypher，不需要写新的查询。
4. 在 `tests/test_query_routing.py::test_candidate_filters` 补断言。

### 加一个路由键

需要同步改这些地方：

1. `domain.ROUTE_SPECS`：证据形态、引擎、来源、validator、关键词。
2. `domain.RULE_SPECS`：规则兜底词，能复用核心词常量就复用。
3. `domain.INTENT_TO_KEYS`：如果它属于新的顶层意图才需要改。
4. `prompts.ANSWER_RULES`：有路由专属回答约束时补。
5. `evidence.py`：`engine="graph_entity"` 需要在 `_graph_entity_docs` 加分支；
   新引擎还要在 `execute` 的 dispatch 和
   `intelligent_query_router._strategy_for_engine` 登记。
6. `validators.VALIDATORS`：新增 validator 名时注册。
7. `tests/test_query_routing.py`：至少补路由和证据断言，再跑黄金集。

### 加一个槽位

1. `domain.SlotVocabulary` 加词表。
2. `slots.SlotExtractor.normalize/extract` 加归一化和原文抽取。
3. 能用于推荐过滤的，补 `domain.CANDIDATE_FILTERS`。
4. 需要跨轮延续的，加进 `DOMAIN.persisted_slots`。

### 改澄清话术

只改 `DOMAIN.clarify_messages`，不要在代码里写死文案。现有键：
`no_intent / ambiguous / low_confidence / reference_missing / empty_query`。

### 调整阈值或灰度

用 `ROUTE_THRESHOLDS_PATH` 指向候选 JSON，不改代码。先拿黄金集对比
`route_key_accuracy`、`clarify_rate`、`false_clarify`，再切默认值。

### 换一个领域

替换 `DOMAIN`（`DomainConfig`）及其词表、Schema、tips 来源。路由包内部不
直接依赖 cookwise 的具体菜名；但 `prompts.py` 和
`tips_knowledge.py` 的文档格式仍需按新领域调整。

### 改决策日志

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ROUTE_LOG_PATH` | `data/logs/route_decisions.jsonl` | 日志路径 |
| `ROUTE_LOG_ENABLED` | `1` | `0/false/no/off` 关闭 |

文件超过 20 MB 时，把当前文件改名为带时间戳的归档，再继续写新文件。

## 11. 排查“为什么路由错了”

1. 先用 `router.explain_routing_decision(query)` 看意图、路由键、证据画像、
   置信度、阈值和执行计划；它只做分析，不检索、不写日志，也不带会话状态。
   多轮追问要改成 `router.analyze_query(query, session_state=state)`。
2. 需要看模型原始输出时，用 `analysis = router.analyze_query(query)`，
   再看 `analysis.understanding["raw"]`。
3. 再看 `data/logs/route_decisions.jsonl` 的同一条记录。
4. 如果是规则兜底问题，按顺序检查 `domain.RULE_SPECS` 和
   `domain.FOLLOW_UP_RULES`；先命中先返回。

JSONL 主要字段：

```text
session_id / query / standalone_query / route_key / intent / engine
needs_retrieval / response_type / confidence / understanding_source
topic_id / used_history / topic_shift / is_multi_intent
sub_queries / evidence_status / sub_evidence / doc_count / latency_ms / error
```

常见现象对照：

| 现象 | 优先看 |
| --- | --- |
| 路由键不对 | `understanding_source`、`raw`、`RULE_SPECS` 顺序 |
| 该检索却没检索 | `needs_retrieval`、对应路由键的阈值 |
| 一直澄清 | `response_type=clarify`，看是 `no_intent / ambiguous / low_confidence / reference_missing / empty_query` 哪一类 |
| 回答没证据 | `evidence_status`、`sub_evidence[].status`、`engine` |
| 多意图串味 | `sub_queries` 的 `route_key`、`query`、`entities`，再看文档的 `sub_query_index` |
| 偶发整条降级 | `error` 字段和后端日志里的“查询路由失败，降级到传统检索” |

## 12. 回归命令

```bash
# 离线单测（不依赖 Neo4j / Milvus / LLM）
python -m pytest -q tests/test_query_routing.py

# 黄金集：本地脚本，需 Neo4j / Milvus / LLM
python scripts/eval/eval_routing.py --limit 20
python scripts/eval/eval_routing.py --workers 4

# 真实多轮 E2E：需后端已启动
python tests/e2e/multiturn_e2e.py

# 图检索链路验证
python tests/test_graph_rag_verification.py
```

`scripts/eval/`、`tests/e2e/`、`tests/gold/` 是本地工具目录，默认不进仓库。

改动涉及路由行为时，至少跑单测 + 黄金集；涉及多轮、生成或流式时再跑 E2E。
如果改了阈值、Prompt 或规则顺序，必须对比改动前后的黄金集报告，不能只看单条案例。
