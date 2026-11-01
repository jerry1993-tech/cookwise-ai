# 路由黄金集与影子流量验收结果

测试日期：2026-10-25
分支：`v01`（未提交）
黄金集：600 条 case / 807 轮；多意图 180 条（30%）、多轮 180 条（30%）、边界 120 条（20%）
12 个路由键覆盖 57~97 次

## 1. 最终黄金集指标

最终代码跑完 600 条 case，汇总在 `eval_report.json`；逐条预测可用
`scripts/eval/eval_routing.py` 重新生成（本轮生成的 `predictions.jsonl` 已清理）。

| 指标 | 结果 |
| --- | --- |
| 路由键准确率 | **1.0000** |
| 路由键 micro precision / recall | **1.0000 / 1.0000** |
| 是否需要检索准确率 | **1.0000** |
| response_type 准确率 | **1.0000** |
| 证据命中率（646 个检查轮） | **1.0000** |
| 槽位准确率（103 个检查轮） | 0.9806 |
| 澄清率 / 期望澄清率 | 0.0558 / 0.0558 |
| 误澄清 / 漏澄清 | 0 / 0 |
| 路由耗时 p50 / p95 | 1490 ms / 2801 ms |

按分类拆分的路由准确率均为 1.0；单轮单意图 300 轮、单轮多意图 120 轮、
多轮单意图 267 轮、多轮多意图 120 轮。

最后一轮修复前后对比：

| 版本 | 路由准确率 | response_type | 证据命中率 | 遗留问题 |
| --- | --- | --- | --- | --- |
| run6 | 0.9988 | 0.9988 | 0.9985 | “我刚开始学做菜”误判推荐；无骨鸡爪实体类型标错导致营养证据缺失 |
| run7 | **1.0000** | **1.0000** | **1.0000** | 仅剩 2 个“半成品”槽位写法差异 |

## 2. 阈值校准

`calibrate_thresholds.py` 用 70/30 切分回放 run7 的理解结果：

- 训练 577 轮，留出 230 轮
- 默认阈值与校准阈值在训练/留出集上均为 route=1.0、needs=1.0、false_clarify=0、missed_clarify=0
- 逐路由键坐标下降后 **0 项变更**，`calibrated_thresholds.json` 与默认值一致

说明：按路由键配置的能力已经落地（`ROUTE_THRESHOLDS_PATH` 可加载 13 个键），
但在当前黄金集上默认三阈值已经最优，没有为了“必须有改动”而强行调参。

## 3. 影子流量与 Canary

同一批 40 条抽样 case（共 54 轮）分别打两个后端：

| 指标 | 基线（8000，默认阈值） | Canary（8001，校准阈值） |
| --- | --- | --- |
| 轮数 | 54 | 54 |
| 请求错误率 | 0.0% | 0.0% |
| 路由键准确率 | 1.0 | 1.0 |
| 是否需要检索准确率 | 1.0 | 1.0 |
| response_type 准确率 | 1.0 | 1.0 |
| 证据缺失率 | 0.0% | 0.0% |
| 证据部分命中率 | 5.56% | 5.56% |
| 回答关键词命中率 | 1.0 | 1.0 |
| 客户端 p95 延迟 | 4897 ms | 4912 ms（基线的 1.003 倍） |
| 服务端 p95 延迟 | 2737 ms | 2987 ms |

回滚阈值均未触发：错误率 0% < 2%；证据缺失 0% < 15%；p95 1.003 倍 < 1.5 倍。
Canary 窗口没有真实用户反馈，点踩率一项不适用（N/A）。

证据“部分命中”的 3 轮都是菜谱原文没有标注热量，系统按 `no_data` 正确回答，
不是检索失败。

Canary 验证完成后已恢复 Nginx 原始配置并 reload：

- `git diff -- nginx/nginx.conf` 为空
- 容器内外配置 md5 均为 `2e1d8c62e87c2ccf543c64488d54a5ab`
- `nginx -T` 中不再出现 `backend_canary` / `X-Canary`
- `/health`、`/api/stats` 均返回 200

## 4. 在线抽查

最终 8000 后端 + 恢复后的 Nginx 上抽查：

- “请问水煮鱼需要哪些食材？” -> `recipe_ingredients`，`evidence_status=complete`，图谱食材 10/10
- “可乐鸡翅的做法是什么？” -> `recipe_overview`，`evidence_status=complete`，
  回答只给图谱里有依据的概要，未编造食材清单或步骤

## 5. 回归门

| 检查 | 结果 |
| --- | --- |
| 多轮真实 E2E（`tests/e2e/multiturn_e2e.py`） | PASS 24 / FAIL 0 |
| Graph RAG 88 项断言 | PASS 88 / FAIL 0 / WARN 0 |
| 离线路由单测 | 24 passed |
| 追问/澄清配置化后全量复跑 | 600 case / 807 轮，质量指标与上表逐项一致 |
| `compileall` | 通过 |
| `validate_gold.py` | 校验通过 |
| 反馈接口模拟（like/dislike/correction） | 200，统计正常 |

`rag_modules/graph_rag_retrieval.py` 未改动；签名检查与 13 查询基线未触发，
88 项断言作为回归检查通过。

## 6. 已知边界

1. 黄金集是按项目真实实体合成的，开发过程中针对失败样例补过 prompt 示例，
   100% 不能替代真实用户验收；建议按 `ANNOTATION.md` 抽检 50~100 条。
2. 影子流量是同黄金集抽样回放，不是独立真实流量；canary 窗口也没有真实用户访问。
3. 槽位准确率 0.9806 来自 2 条“推荐几道半成品”的槽位写法差异
   （LLM 写成 `keyword`/`tags`，期望 `category`），路由与证据均正确。
4. 答案忠实度目前用“证据命中 + 回答关键词命中”做确定性代理，
   没有引入 LLM judge；后续可加。
5. CI 依赖 `tests/gold/golden_set.jsonl` 与 `entities.json` 入库；
   `predictions.jsonl`、`shadow_*_turns.jsonl` 是可由脚本重新生成的中间产物，本轮已清理。

## 7. 辣度/食材/否定偏好修复后复跑

在 run7 基础上，补了通用候选过滤（`CANDIDATE_FILTERS`）、辣度与口味槽位归一化、
食材过滤走 `(:Recipe)-[:REQUIRES]->(:Ingredient)`、以及多轮否定偏好累计；
`graph_rag_retrieval.py` 未改动。

| 检查 | 结果 |
| --- | --- |
| 黄金集 | 600 case / 807 轮 |
| 路由键 / 是否需要检索 / response_type / 证据命中率 | **1.0000 / 1.0000 / 1.0000 / 1.0000** |
| 槽位准确率（103 轮） | 0.9806（仍为 2 条“半成品”写法差异） |
| 澄清率 / 期望澄清率 / 误澄清 / 漏澄清 | 0.0558 / 0.0558 / 0 / 0 |
| 路由耗时 p50 / p95 | 1747 ms / 6766 ms |
| 多轮真实 E2E | **PASS 24 / FAIL 0** |
| 离线路由单测 | **27 passed** |
| Graph RAG 88 项断言 | PASS 88 / FAIL 0 / WARN 0 |
| `compileall` / `validate_gold.py` | 通过 / 校验通过 |

关键词兜底切分改为 jieba 后再次全量复跑：600 case / 807 轮，
路由 / needs / response_type / 证据命中率仍为 1.0，槽位 0.9806，澄清率 0.0558；
本轮日志未触发“关键词提取失败”兜底，质量指标与上表一致（耗时 p95 9271 ms 属外部 LLM 波动）。

在线抽查（8000 最终后端）：

- “那哪些菜是带辣椒的” -> 8 道含辣椒食材的菜，并显示“命中食材”
- “推荐几道带辣椒的川菜” -> 食材 + 菜系叠加过滤，候选为川菜且带辣椒
- “有哪些不辣的菜” -> 8 道不辣候选
- “我不喜欢甜品” -> “推荐几道菜” -> 微波炉蛋糕被排除
- “推荐几道带牛肉的菜” / “推荐几道甜口的菜” / “素菜，不要香菜” -> 过滤生效

本轮 `predictions.jsonl` 已清理；代码仍未提交。

## 8. 可维护性重构后复跑（2026-10-26）

目标：减少重复词表、单文件职责和路由键分发点，不改变路由行为。

| 检查 | 结果 |
| --- | --- |
| 离线路由单测 + Web 冒烟 | **30 passed + 1 passed** |
| 黄金集 | **600 case / 807 轮** |
| 路由键 / needs / response_type / 证据命中率 | **1.0000 / 1.0000 / 1.0000 / 1.0000** |
| 路由键 micro precision / recall | **1.0000 / 1.0000** |
| 槽位准确率（103 轮） | 0.9806（仍为 2 条“半成品”写法差异） |
| 澄清率 / 期望澄清率 / 误澄清 / 漏澄清 | 0.0558 / 0.0558 / 0 / 0 |
| 路由耗时 p50 / p95 | 1372 ms / 3227 ms（LLM 外部波动，不作为性能承诺） |
| 多轮真实 E2E | **PASS 24 / FAIL 0** |
| Graph RAG 88 项断言 | **PASS 88 / FAIL 0 / WARN 0** |
| `test_router.py` / `compileall` / `test_web.py` | 通过 / 通过 / 通过 |

本轮重构内容：

- `understanding.py` 拆出 `slots.py`（槽位归一化/抽取）和 `rules.py`（规则分类/追问/意图合并），从 1,107 行降到 716 行。
- `domain.py` 把 route / rule 共用的核心词抽成常量；规则词表与 Prompt 扩展词分开，词表集合保持不变。
- LLM 只输出语义字段；`evidence_profile` / `need_complete` / `value_policy` / `expected_source` 由 `RouteKeySpec` 派生，`intent` 以 `route_key` 对应配置为准。
- 回答约束集中到 `prompts.ANSWER_RULES`；跨轮槽位集中到 `DomainConfig.persisted_slots`。
- `evidence.py` 的候选查询拆成“收集条件”和“拼 Cypher”两段；补充 graph_entity 路由分发覆盖测试。
- 清理 `EvidenceResult`、`_combined_search`、不可达分支；路由统计改为字典映射。
- 新增 `rag_modules/query_routing/README.md` 维护地图，记录常见改动和回归命令。

说明：黄金集与 Graph RAG 验证均通过；后端当前运行的是本轮最终代码，`graph_rag_retrieval.py` 未改动，代码仍未提交。

## 9. 混合检索对齐官方实现（BM25 + RRF，2026-10-26）

改动范围只有 `hybrid_retrieval.py` 和新增 `tests/test_hybrid_fusion.py`，
不涉及路由、生成和 `graph_rag_retrieval.py`。

| 检查 | 结果 |
| --- | --- |
| 黄金集 | **600 case / 807 轮** |
| 路由键准确率 | 0.9988（1 条多意图 LLM 判错，单独复跑 3 次全对，见下） |
| 路由键 micro precision / recall | 0.9989 / 0.9989 |
| needs / response_type | **1.0000 / 1.0000** |
| 证据命中率 | 0.9985（与上面同一条 case 相关） |
| 槽位准确率（103 轮） | 0.9806（仍为 2 条“半成品”写法差异） |
| 澄清率 / 期望澄清率 / 误澄清 / 漏澄清 | 0.0558 / 0.0558 / 0 / 0 |
| 路由耗时 p50 / p95 | 1935 ms / 7419 ms（外部 LLM 波动，不作为性能承诺） |
| 多轮真实 E2E | **PASS 24 / FAIL 0** |
| 离线单测（路由 + Web + 混合检索） | **40 passed** |

对齐内容：

- **BM25**：`rank_bm25.BM25Okapi` + `jieba.lcut` + 中文停用词表；`bm25_score` 写入 metadata，
  分数 ≤ 0 的结果丢弃。实跑索引 1138 条 chunk、平均 111.5 token。
- **RRF**：`_rrf_merge` 静态方法，输入 `[(source_name, ranked_docs), ...]`，
  `score = Σ 1/(60 + 该路最好名次)`；同一路里同一道菜的多个 chunk 只算一次，
  命中次数记进 `rrf_chunk_hits`；正文取全局最好名次那条；返回新 Document，不改上游对象。
- **融合入口**：三路统一 `candidate_k = max(top_k * 2, 10)`，避免各路前 top_k 没有交集。
- **父文档回填**：`_build_parent_doc_map` / `_attach_parent_documents`，
  由 `enable_parent_doc_retrieval` 控制（本轮已在 `config.py` 打开，`parent_doc_top_n=3`、
  `parent_doc_max_chars=4000`）。

与官方代码的有意差异：

1. 三路检索用线程池并行执行；融合输入和结果与串行一致，只是少等时间。
2. 关键词提取失败走项目的图谱词典 + jieba 兜底，而不是官方示例里的 `query.split()`（中文无效）。
3. LLM 调用带 `extra_body={"thinking": {"type": "disabled"}}`，避免推理模型吃光 max_tokens。
4. `RetrievalResult.retrieval_level` 注释改为 `'entity' or 'topic'`（官方注释写 low/high，与实际值不符）。

失败用例说明：`mi-normal-074`「腊八粥先做什么后做什么？另外，简易版炒糖色难不难做？」
第二个子问题被判成 `recipe_steps_time`（期望 `recipe_overview`），该轮 engine 是 `graph_entity`，
不经过本次改动的混合检索；同一 case 单独复跑 3 次全部正确，判定为 LLM 输出波动。

说明：`graph_rag_retrieval.py` 未改动，88 项断言本轮未重跑；代码仍未提交。

### 9.1 打开父文档回填后复跑

在 `config.py` 加 `enable_parent_doc_retrieval=True`、`parent_doc_top_n=3`、
`parent_doc_max_chars=4000` 后复跑（会改变生成层看到的上下文，故重跑质量指标）：

| 检查 | 结果 |
| --- | --- |
| 父文档映射 | 323 篇菜谱 |
| 实机抽查 | `红烧肉的做法` 前 3 条为整篇菜谱（716/753/1012 字），第 4 条仍为原 chunk；食材文档不受影响 |
| 黄金集 | **600 case / 807 轮** |
| 路由键 / needs / response_type / 证据命中率 | **1.0000 / 1.0000 / 1.0000 / 1.0000** |
| 路由键 micro precision / recall | **1.0000 / 1.0000** |
| 槽位准确率（103 轮） | 0.9806（仍为 2 条“半成品”写法差异） |
| 澄清率 / 期望澄清率 / 误澄清 / 漏澄清 | 0.0558 / 0.0558 / 0 / 0 |
| 路由耗时 p50 / p95 | 1908 ms / 6309 ms |
| 多轮真实 E2E | **PASS 24 / FAIL 0** |
| 离线单测 | **40 passed** |

本轮只改 `config.py`（加 3 个键并写入 `to_dict`），检索与路由代码未动。

## 10. 混合检索按职责拆分（2026-10-26）

原 `hybrid_retrieval.py` 848 行、一个类承担装配 + 四类检索 + 融合 + 上下文处理。
本轮只做代码搬迁与归类，不新增业务逻辑、不改检索行为：

| 文件 | 职责 | 行数 |
| --- | --- | --- |
| `rag_modules/hybrid_retrieval.py` | 门面：装配、三路并行 → RRF → 父文档回填 | 215 |
| `hybrid_rag_retrieval_components/tokenizer.py` | 中文分词 + 停用词 | 25 |
| `hybrid_rag_retrieval_components/bm25_index.py` | BM25 建索引 / 检索 | 79 |
| `hybrid_rag_retrieval_components/graph_layer.py` | 图索引构建、Neo4j 邻居查询 | 88 |
| `hybrid_rag_retrieval_components/dual_level_retriever.py` | 关键词提取 + 实体级 / 主题级检索 | 423 |
| `hybrid_rag_retrieval_components/vector_retriever.py` | 向量检索 + 邻居增强 | 67 |
| `hybrid_rag_retrieval_components/fusion.py` | RRF 融合 | 96 |
| `hybrid_rag_retrieval_components/parent_documents.py` | 父文档映射与回填 | 50 |

目录原名 `hybrid_retrieval_parts/`，2026-10-27 改名为 `hybrid_rag_retrieval_components/`，
与图 RAG 的 `graph_rag_retrieval_components/` 命名保持一致；仅改目录名与 import 路径，代码内容未动。

对外接口保持不变（`HybridRetrievalModule`、`RetrievalResult`、全部公开方法与
`bm25` / `bm25_corpus_docs` 属性），`rag_modules/__init__.py` 与上层调用均无需修改。

| 检查 | 结果 |
| --- | --- |
| 离线单测（路由 + Web + 混合检索） | **40 passed** |
| 实机对照（拆分前后同 query） | 结果逐条一致：同 node_id、同顺序、同正文长度 |
| 黄金集 | **600 case / 807 轮** |
| 路由键 / needs / response_type / 证据命中率 | **1.0000 / 1.0000 / 1.0000 / 1.0000** |
| 槽位准确率（103 轮） | 0.9806（仍为 2 条“半成品”写法差异） |
| 多轮真实 E2E | **PASS 24 / FAIL 0** |
| `compileall` | 通过 |

说明：`graph_rag_retrieval.py` 未改动，代码仍未提交。

## 11. 图 RAG 检索按职责拆分（2026-10-27）

原 `graph_rag_retrieval.py` 4,118 行，一个文件承载 Neo4j 访问、实体索引、查询理解、
路径遍历、子图抽取、推理、文档渲染与排序。本轮只做代码搬迁与归类，
不新增业务逻辑、不新增工具方法、不改检索行为；方法体逐字保留。

| 文件 | 职责 | 行数 |
| --- | --- | --- |
| `rag_modules/graph_rag_retrieval.py` | 门面：原模块说明、组件装配、共享状态初始化、主检索流程 | 246 |
| `graph_rag_retrieval_components/shared_constants_helpers_models.py` | 共享底座：图结构常量、模块级纯函数、`QueryType` / `GraphQuery` / `GraphPath` / `KnowledgeSubgraph` | 422 |
| `graph_rag_retrieval_components/neo4j_store_access.py` | Neo4j 连接生命周期、只读 Cypher、节点/关系序列化、度数补齐、关系去重 | 401 |
| `graph_rag_retrieval_components/entity_name_index.py` | 图索引预热、实体名规范化、别名扩展、候选实体抽取、种子节点解析 | 579 |
| `graph_rag_retrieval_components/query_understanding.py` | 自然语言 → `GraphQuery`（LLM + 本地降级）、复杂度分析、自适应规划 | 365 |
| `graph_rag_retrieval_components/path_retrieval_engine.py` | `GraphPath` 构建与评分、多跳遍历、实体关系 / 最短路径 / 相似实体查询 | 935 |
| `graph_rag_retrieval_components/subgraph_extraction_and_reasoning.py` | 两阶段子图抽取与降级、拓扑推理链构造与校验 | 916 |
| `graph_rag_retrieval_components/document_formatting_and_ranking.py` | 路径/子图渲染为 `Document`、图结构分、四分量融合排序 | 427 |
| `graph_rag_retrieval_components/__init__.py` | 组件导出 | 41 |

目录名用 `graph_rag_retrieval_components/` 而不是 `graph_rag_retrieval/`：后者会和
`graph_rag_retrieval.py` 同名，Python 导入时包会遮蔽模块。

对外接口保持不变：`GraphRAGRetrieval` 仍是十个 mixin 的门面，`__bases__` 顺序、66 个方法与签名、
四个数据结构的字段 / 枚举成员、`rag_modules/__init__.py` 的惰性导出全部未变；`main.py`、
路由层与测试脚本无需修改。

| 检查 | 结果 |
| --- | --- |
| `py_compile` / `compileall` | 通过 |
| 原文逐字比对（10 段方法体 + 模块说明 + 门面类体） | **IDENTITY: OK**（无手工重写） |
| 模块级未定义名字（AST 检查 8 个文件） | **UNDEFINED: OK** |
| 基类顺序 / 方法数 / 签名 / 数据模型 | **OK / 66 = 66 / OK / OK** |
| 13 查询基线（拆分前 vs 拆分后，固定 LLM 返回 `{}`） | **13 条 diffs=0**（查询类型、实体、深度、每条结果的 node_id / 分数 / 正文全等） |
| Graph RAG 88 项断言 | **PASS 88 / FAIL 0 / WARN 0** |
| 离线单测（路由 + Web + 混合检索） | **40 passed** |
| 黄金集 | **600 case / 807 轮** |
| 路由键 / needs / response_type / 证据命中率 | **1.0000 / 1.0000 / 1.0000 / 1.0000** |
| 路由键 micro precision / recall | **1.0000 / 1.0000** |
| 槽位准确率（103 轮） | 0.9806（仍为 2 条“半成品”写法差异） |
| 澄清率 / 期望澄清率 / 误澄清 / 漏澄清 | 0.0558 / 0.0558 / 0 / 0 |
| 路由耗时 p50 / p95 | 1486 ms / 3513 ms |
| 多轮真实 E2E（后端已重启加载新代码） | **PASS 24 / FAIL 0** |

说明：本轮只有文件搬迁，检索结果与拆分前逐条一致；代码仍未提交。

## 12. 多意图回答编号修复（2026-10-27）

问题：多意图回答出现两个 `1.`（实测 `空气炸锅面包片具体怎么做？需要什么材料？`、
`麻辣减脂荞麦面需要什么材料？具体怎么做？`），序号不连续。

定位：

- 路由层正常：`is_multi_intent=true`，每轮两个子查询、两路证据都是 `complete`；
  日志里只有一次生成调用，排除“后端拼接两段答案”。
- 根因在生成 Prompt 与生成方式：子问题清单只是 Prompt 里的一段文字，编号是否连续完全依赖模型；
  两个带 `子问题：…` 标签的证据块又强化了“两段独立小答案”，模型答到第二段时重新从 `1.` 开始。
- 只补 Prompt 约束（第一版修复）实测仍会复现：模型在长回答里不保证遵守编号规则。

最终方案：**编号不再交给模型，由代码负责回答结构。**

- `_multi_intent_sections` 按文档的 `sub_query_index` 把证据分到各子问题；可安全拆分时，
  每个子问题单独生成一节，节标题与 `1./2./3.` 全部由代码写出，模型只负责该节正文。
- 每节使用该子问题的 `route_key` 应用各自的 `ANSWER_RULES`；免责声明只加在最后一节。
- 模型若仍复述子问题标题，`_strip_section_heading` 确定性地去掉该首行，避免重复标题与编号。
- 节标题写成 markdown 标题块 `### 1. …`，而不是裸的 `1. …`：前端用 react-markdown 渲染，
  裸编号会被当成有序列表项，与正文里的步骤列表合并（步骤从 2 开始、第二种回答被并成第 10 项）。
  标题块会终止列表，各节的列表重新从 1 开始。
- 无法安全拆分时回退原来的单次生成：非多意图、有效子问题少于 2 个、子问题存在 `depends_on`、
  证据未按子问题打标，或某个子问题没有证据；上一版 Prompt 约束保留为该回退路径的兜底。
- 单意图路径不变；多意图由 token 级流式改为“逐节生成后输出”，以结构正确性优先。

第二轮线上又发现：分节后的第二段变成一整段流水账（`水煮鱼需要什么食材？怎么做呀？`）。定位与改法：

- 根因是上一轮加的 section 规则“直接回答该问题，不要重复问题原文或添加标题”被模型理解成“不要分条”，
  8 个步骤被合并成一段。对照实验：同一份步骤证据下删掉该规则，回答立刻恢复 `1..8` 编号。
- 删除该规则；对“怎么做/做法/步骤”类问句，在 Prompt 里明确“每一步单独一行并编号”；问时间不加，
  避免“要煮多久”被展开成整套步骤。
- 加确定性兜底 `_ensure_step_structure`：证据里是图谱步骤（≥3 步）且回答没有分条时，按证据重建编号列表，
  结构不再取决于模型是否听话；非 how-to 问句不触发。

| 检查 | 结果 |
| --- | --- |
| 离线单测（路由 + Web + 混合检索） | **42 passed** |
| 「麻辣减脂荞麦面需要什么材料？具体怎么做？」实机 | 顶层 `1.` 材料、`2.` 做法，连续 |
| 「空气炸锅面包片具体怎么做？需要什么材料？」实机 | 顶层 `1./2.` 连续（节顺序由路由给出的子问题顺序决定） |
| 「水煮鱼需要哪些食材？要煮多久？有什么营养？」实机 | 顶层 `1./2./3.` 连续，免责声明只出现一次 |
| 新增测试 `test_multi_intent_answer_numbering_is_code_owned` | 每节只拿到自己的证据；模型复述的标题被去除；编号只由代码写一次；节标题是标题块而非列表项（同步/流式都断言） |
| 新增测试 `test_steps_answer_structure_is_repaired` | 流水账按证据重建为 1./2./3.；已分条原样返回；“要煮多久”不展开；证据不足 3 步不重建 |
| 「水煮鱼需要什么食材？怎么做呀？」实机 | 第 2 节为 8 条编号步骤，不再合并成一段 |
| 「水煮鱼要煮多久？」实机 | 仍按时间点简要回答，未展开成整套步骤 |
| 多轮真实 E2E | **PASS 24 / FAIL 0** |

本轮改生成层（`rag_modules/generation_integration.py`）与对应测试；路由、检索、证据层未动，
因此黄金集未重跑。代码仍未提交。

两轮修复后做过一次等价压缩（只精简注释、去掉冗余判断、把 42 行 Prompt 段落收成兜底约束），
行为不变：前两轮生成层累计 **+99 / −16（净 +83）**，测试 +62 行；压缩前后离线 41 passed、
实机与流式编号连续、E2E 均 PASS 24 / FAIL 0。第三轮（结构兜底）在此基础上的增量未单独统计，
当前离线为 42 passed。
