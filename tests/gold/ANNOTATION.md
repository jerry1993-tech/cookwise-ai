# 黄金集标注口径

本目录的黄金集由 `scripts/eval/generate_gold.py` 从项目真实数据合成：

- 菜名、食材、步骤、热量、营养描述：Neo4j + `data/` 下 323 篇菜谱原文
- 相克、技法、食品安全、配菜规则：`data/` 下 tips 文档
- 域外问题不依赖项目数据，仅作为负例

共 600 条 case，比例固定为多意图 30%、多轮 30%、边界 20%，12 个路由键各 30~100 次。
人工抽检时按下面的口径判断即可。

## 文件结构

`golden_set.jsonl` 每行一条 case：

```json
{
  "id": "base-recipe_ingredients-001",
  "category": "single_turn_single_intent",
  "boundary": false,
  "multi_intent": false,
  "multi_turn": false,
  "turns": [
    {
      "query": "芝麻烧饼需要哪些食材？",
      "expected": {
        "route_keys": ["recipe_ingredients"],
        "needs_retrieval": true,
        "response_type": "answer",
        "evidence": [
          {
            "route_key": "recipe_ingredients",
            "keywords": ["食用油"],
            "recipe_names": ["芝麻烧饼"],
            "match": "all"
          }
        ],
        "slots": {}
      }
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `category` | 四个组合：单轮单意图 / 单轮多意图 / 多轮单意图 / 多轮多意图 |
| `boundary` | 是否属于边界 case（域外、闲聊、澄清、域外+正常多意图） |
| `route_keys` | 本轮应该命中的路由键；多意图按集合比较，顺序不重要 |
| `needs_retrieval` | 本轮是否需要检索。闲聊/域外单轮为 `false`，与菜谱任务组合时为 `true` |
| `response_type` | `answer` / `clarify` / `refuse` / `no_data` |
| `evidence` | 证据检查项，详见下节；闲聊、域外、澄清轮为空 |
| `slots` | 推荐类查询的过滤条件；只有推荐类才声明 |

`entities.json` 是生成时的实体快照，供离线校验和人工核对，不参与在线评测。

## 12 个路由键

| 路由键 | 触发问法（示例） | 期望证据 |
| --- | --- | --- |
| `recipe_ingredients` | “X 需要哪些食材/原料/调料？” | 菜名 + 该菜真实存在的一种食材 |
| `recipe_steps_time` | “X 怎么做/步骤是什么/要多久/几分钟/火候？” | 菜名 + 原文烹饪时间（如“1分钟”） |
| `recipe_overview` | “X 的做法是什么/是什么菜/介绍一下/有什么特点/哪个菜系？” | 菜名 + 分类或菜系 |
| `ingredient_pairing` | “A 和 B 能一起吃吗/同食/相克/禁忌？” | tips 相克条目中的两个食材 |
| `ingredient_property` | “X 怎么去腥/焯水/腌制/凉拌？能用微波炉/空气炸锅/高压锅吗？” | 技法词（去腥、焯水、腌制、凉拌、微波炉、空气炸锅、高压锅） |
| `recommend_list` | “推荐几道素菜/川菜/快手菜/低热量的菜” | 菜类、菜系、标签或“候选菜谱” |
| `recommend_scenario` | “四个人吃点什么/一桌菜怎么配/加条鱼/荤素怎么搭？” | “配菜规则” |
| `nutrition_value` | “X 有多少千卡/热量/蛋白质/脂肪？” | 菜名 + 原文热量；原文没有时为 `no_data`、无证据 |
| `nutrition_benefit` | “X 有什么营养/功效/好处/适合谁？” | 菜名 + 原文营养词 |
| `food_safety` | “X 会不会中毒/过敏/有寄生虫/孕妇能吃吗？” | “食品安全” + 对应安全段落 |
| `chitchat` | “好累/心情不好/好想吃甜的/你好” | 无，`needs_retrieval=false` |
| `out_of_scope` | 天气、股票、代码、翻译、旅行等非饮食问题 | 无，`needs_retrieval=false` |

## 歧义 case 的拍板口径

1. **“X 的做法是什么？” -> `recipe_overview`**，不展开成步骤。
   只有明确出现“步骤 / 怎么做 / 怎么煮 / 怎么炒 / 多久 / 火候”才归 `recipe_steps_time`。
2. **“热量/蛋白质/脂肪是多少？” -> `nutrition_value`**。
   原文没有精确数值时标 `no_data`，不估算；只有 `nutrition_value` 允许 `no_data`。
3. **相克/同食问题 -> `ingredient_pairing`**，证据来自 tips 文档，不走 Neo4j 关系。
4. **推荐单道菜还是推荐一组菜**：问“推荐几道/有什么菜/吃什么” -> `recommend_list`；
   问人数、荤素搭配、加鱼、一桌菜 -> `recommend_scenario`。
5. **菜类与饮食约束分开**：“素菜/荤菜/汤类/主食/甜品/水产/早餐/凉菜/饮料”放 `category`；
   “减脂/低热量”放 `diet`，两者不混用。
6. **情绪 + 菜谱任务要拆成两个意图**，例如“今天工作好累。顺便问一下，速冻汤圆有多少千卡？”
   -> `chitchat` + `nutrition_value`；纯寒暄“你好”可以忽略，不单独算一项。
7. **域外 + 菜谱任务**保留两个意图，`needs_retrieval=true`，只给菜谱部分标证据。
8. **澄清轮**（指代不明、多个候选菜）`route_keys=[]`、`needs_retrieval=false`、`response_type=clarify`。

## 证据检查

评测时，一条证据必须被**同一份文档**同时满足才记为命中：

- `route_key`：文档的 `sub_route_key` 或 `route_key` 与之相同
- `recipe_names`：所有菜名都出现在正文或元数据里
- `keywords`：`match="all"` 时全部出现，`match="any"` 时任意一个出现
- `forbidden`：正文出现任意禁用词则跳过该文档（仅用于“不辣”这类负向推荐）

`evidence` 为空表示该轮不做证据检查（闲聊、域外、澄清、`no_data`）。
推荐类如果只是“随便推荐几道家常菜”，没有具体菜类/菜系词，`keywords` 可以为空，
表示只要返回任意候选菜谱文档即算命中；有 `forbidden` 时仍要求候选文档不含禁用词。

`nutrition_value` / `nutrition_benefit` 的证据取自菜谱原文的简介或“附加内容”。
当前 Milvus 只索引菜谱主体，若这些段落没有被召回，系统会返回 `no_data` 或部分证据；
这类 case 用来衡量检索覆盖率，属于真实缺口，不建议为了让指标好看而删除。

## 评测指标

`scripts/eval/eval_routing.py` 输出：

- `route_key_accuracy`：每轮预测的路由键集合与标注完全一致的比例（最硬指标之一）
- `route_key_micro_precision/recall`：按路由键展开的微平均
- `needs_retrieval_accuracy`：是否检索判断准确率
- `response_type_accuracy`：answer/refuse/clarify/no_data 判断准确率
- `evidence_hit_rate`：证据命中率（最硬指标之一）
- `slot_accuracy`：单意图推荐类查询的槽位准确率
- `clarify_rate` / `false_clarify` / `missed_clarify`：澄清取舍，业务上尽量少澄清
- `latency_ms_p50/p95`：单轮路由耗时

## 重新生成与校验

```bash
# 需要 Neo4j 可用；Milvus 不需要
python scripts/eval/generate_gold.py
python scripts/eval/validate_gold.py
```

`validate_gold.py` 完全离线，CI 会跑；它检查 ID、比例、路由键覆盖、槽位和证据实体是否自洽。
`validate_gold.py` 通过不代表模型一定答对，线上表现仍以 `eval_routing.py` 为准。

## 人工抽检

建议按 `category` 各抽 15~25 条，一共 50~100 条，重点看：

1. 问法与 `route_keys` 是否符合上面的口径；
2. 多意图是否漏拆或过度拆分；
3. 多轮第二轮是否依赖上一轮实体，标注是否写成了 standalone；
4. `evidence` 的关键词是否确实出现在对应文档；
5. `recommend_list` 的 `slots` 是否是有效的图谱值（不是“家常菜”这类口语词）。
