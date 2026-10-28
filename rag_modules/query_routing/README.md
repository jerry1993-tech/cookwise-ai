# 查询路由维护地图

这份文档只回答三个问题：请求怎么走、常见改动改哪里、出错怎么排查。

## 1. 一次请求的 5 步

```text
query
  → understanding.py  理解层：LLM 结构化输出 + 规则兜底
  → rules.py          规则分类：追问/规则路由/意图合并
  → policy.py         策略层：确定性生成 RoutePlan
  → evidence.py       执行层：graph / text / tips 取证据
  → validators.py     证据校验：complete / partial / missing
  → generation_integration.py 生成回答
```

关键原则：

- LLM 只负责语义理解（`route_key`、`span`、`entities`、`slots`、`confidence`）；
- `RouteKeySpec` 是路由元数据的唯一来源（证据形态、完整性、来源、引擎）；
- 路由业务词表只放在 `domain.py`，`rules.py` / `slots.py` 只做匹配和清洗；
- 跨轮保留哪些槽位由 `DomainConfig.persisted_slots` 决定；
- 任何 LLM 解析失败都收敛到规则兜底，不抛异常到主链路。

## 2. 文件职责

| 文件 | 职责 |
| --- | --- |
| `domain.py` | 路由键、规则、词表、候选过滤、图谱/向量库 Schema |
| `prompts.py` | 理解 Prompt、回答约束 `ANSWER_RULES` |
| `understanding.py` | LLM 调用/解析、实体规范化、上下文、澄清 |
| `rules.py` | 追问分类、规则路由、意图合并、纯约束声明 |
| `slots.py` | 槽位归一化和原文抽取 |
| `policy.py` | 置信度门控、多意图拆分、子查询 |
| `evidence.py` | Neo4j / 菜谱原文 / tips / hybrid 证据 |
| `validators.py` | 证据覆盖校验 |
| `telemetry.py` | 路由决策 JSONL |
| `tips_knowledge.py` | CookWise tips 文档解析 |

## 3. 常见改动

### 加一个同义词

1. 优先加到 `domain.py` 的核心词常量（如 `FOOD_SAFETY_TERMS`），`ROUTE_SPECS` 和 `RULE_SPECS` 会同时生效。
2. 如果只是 Prompt 示例/扩展词，改 `ROUTE_SPECS` 对应 `keywords`；再确认 `RULE_SPECS` 是否需要单独加。
3. 跑 `python -m pytest -q tests/test_query_routing.py`。

### 加一个推荐过滤维度

1. 在 `domain.CANDIDATE_FILTERS` 加一条 `CandidateFilterSpec`。
2. `fields` 可选 `name / tags / category / cuisine / ingredient`。
3. `evidence.py` 会按配置生成 Cypher，不需要写新的查询。
4. 在 `tests/test_query_routing.py::test_candidate_filters` 补断言。

### 加一个路由键

需要同步改这些地方：

1. `domain.ROUTE_SPECS`：路由定义、证据形态、引擎、validator。
2. `domain.RULE_SPECS`：规则兜底词（能复用核心词常量就复用）。
3. `prompts.ANSWER_RULES`：如果有路由专属回答约束。
4. `evidence._graph_entity_docs`：只有 `engine="graph_entity"` 才需要。
5. `validators.VALIDATORS`：新增 validator 名时注册。
6. `tests/test_query_routing.py`：至少补路由/证据断言；再跑黄金集。

### 排查“为什么路由错了”

1. 先用 `router.explain_routing_decision(query)` 看理解结果、阈值和执行计划。
2. 再看 `data/logs/route_decisions.jsonl` 里的 `route_key`、`engine`、`evidence_status`。
3. 如果是 LLM 输出问题，先看 `understanding.raw` 里的原始 JSON。
4. 如果是规则兜底问题，直接查 `domain.RULE_SPECS` 的匹配顺序。

## 4. 回归命令

```bash
python -m pytest -q tests/test_query_routing.py
python scripts/eval/eval_routing.py --workers 4
python tests/e2e/multiturn_e2e.py
python tests/test_graph_rag_verification.py
```

改动涉及路由行为时，至少跑单测 + 黄金集；涉及多轮/生成时再跑 E2E。
