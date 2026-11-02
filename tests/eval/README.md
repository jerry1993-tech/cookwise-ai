# 路由评测与阈值校准

黄金集、评测、校准、影子回放的完整流程。标注口径见
[`tests/gold/ANNOTATION.md`](../gold/ANNOTATION.md)。

## 0. 前置

- Neo4j / Milvus 可用（生成黄金集只需要 Neo4j；在线评测两者都要）
- 后端用本机 conda 环境：`conda activate cookwise-ai`
- 首次跑评测前设置离线变量，避免卡在 huggingface.co：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

## 1. 生成与校验黄金集

```bash
python tests/eval/generate_gold.py
python tests/eval/validate_gold.py
```

| 产物 | 说明 |
| --- | --- |
| `tests/gold/golden_set.jsonl` | 600 条 case，多意图/多轮/边界 = 30%/30%/20% |
| `tests/gold/entities.json` | 生成时的实体快照，供离线校验 |
| `tests/gold/ANNOTATION.md` | 标注口径，人工抽检按它判断 |

`validate_gold.py` 不连数据库，CI 会跑。生成脚本可重复执行，ID 和内容稳定。

## 2. 跑全量评测

```bash
python tests/eval/eval_routing.py --workers 4
```

进程内初始化完整系统，按生产路径调用 `route_query` / `update_state_after_turn`，
每条 case 用独立 session。首次初始化约 3~5 分钟，600 条约 10~15 分钟（4 并发）。

| 产物 | 说明 |
| --- | --- |
| `tests/gold/eval_report.json` | 总体/分类/分路由键指标 |
| `tests/gold/predictions.jsonl` | 每轮预测和完整 understanding，校准输入 |

关键指标：

- `route_key_accuracy`、`route_key_micro_precision/recall`（最硬）
- `evidence_hit_rate`（最硬）
- `needs_retrieval_accuracy`、`response_type_accuracy`
- `clarify_rate` / `false_clarify` / `missed_clarify`
- `latency_ms_p50/p95`

`--limit N` 只跑前 N 条，`--workers N` 控制并发；并发过高可能触发 LLM 限流。

## 3. 按路由键校准阈值

```bash
python tests/eval/calibrate_thresholds.py
```

输入 `predictions.jsonl`，用生产 `RoutePolicy` 回放不同阈值，不重新调用 LLM：

- 按 case 70/30 切分，多轮不会跨训练/留出集
- 坐标下降，每个路由键只调会影响它的字段：
  - 检索类：`min_execute`、`retrieval_override`
  - 闲聊/域外：`no_retrieval`
- 目标顺序：路由正确 > 检索判断正确 > 少澄清
- 校准后训练集整体变差时自动回退默认值

| 产物 | 说明 |
| --- | --- |
| `tests/gold/calibrated_thresholds.json` | 可直接给 `ROUTE_THRESHOLDS_PATH` |
| `tests/gold/calibration_report.json` | 默认 vs 校准、训练/留出对比、逐键改动 |

用校准后的阈值重跑评测：

```bash
ROUTE_THRESHOLDS_PATH=tests/gold/calibrated_thresholds.json \
  python tests/eval/eval_routing.py --workers 4 \
  --out tests/gold/eval_report_calibrated.json \
  --predictions tests/gold/predictions_calibrated.jsonl
```

只有留出集没有变差、且业务上接受更高/更低的澄清率时，才把阈值写进
`data/config/route_thresholds.json` 之类的固定配置；否则保持默认。

## 4. 影子回放（无真实流量时的替代）

```bash
# 终端 A：后端（8000）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py

# 终端 B：回放
python tests/eval/shadow_replay.py --limit 20
python tests/eval/shadow_replay.py --sample 40   # 按固定间隔抽样，覆盖四个分类
python tests/eval/shadow_replay.py               # 全量
```

每轮请求后从 `data/logs/route_decisions.jsonl` 取回真实决策，与黄金集对比。
后端已有路由落盘，无需额外埋点；回放用独立 session，不污染真实会话。

| 产物 | 说明 |
| --- | --- |
| `tests/gold/shadow_turns.jsonl` | 逐轮 query / 期望 / 预测 / 证据状态 / 延迟 |
| `tests/gold/shadow_report.json` | 错误率、空答率、路由准确率、证据缺失率、p95 |

回放串行执行，保证日志按 session + query 能准确对上；全量约 30~60 分钟。

### 反馈闭环

后端新增两个接口，落盘到 `data/logs/feedback.jsonl`（可用 `FEEDBACK_LOG_PATH` 覆盖）：

- `POST /api/feedback`：`{"session_id", "query", "answer", "rating": "like|dislike", "correction"}`
- `GET /api/feedback/stats`：点赞/点踩/纠错计数

没有前端入口时可以先模拟：

```bash
python tests/eval/simulate_feedback.py --limit 6
```

## 5. Canary（临时）

1. 用校准阈值在 8001 起候选后端：

```bash
WEB_PORT=8001 \
ROUTE_THRESHOLDS_PATH=tests/gold/calibrated_thresholds.json \
ROUTE_LOG_PATH=data/logs/route_decisions_canary.jsonl \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py
```

2. Nginx 用 `split_clients` 把一小部分 `/api/` 流量分到 8001（容器里用
   `host.docker.internal` 或 `host-gateway` 的宿主机地址）。

3. 观察至少一个落点：`data/logs/route_decisions_canary.jsonl` 或 `/api/stats`。
   没有真实流量时，用同一个抽样集分别打主后端和 canary，指标可直接对比：

```bash
# 基线：直连 8000
python tests/eval/shadow_replay.py --base-url http://localhost:8000 \
  --sample 40 --route-log data/logs/route_decisions_backend.jsonl \
  --out tests/gold/shadow_baseline_turns.jsonl \
  --report tests/gold/shadow_baseline_report.json

# canary：经 Nginx，用请求头强制进 8001
python tests/eval/shadow_replay.py --base-url http://localhost \
  --sample 40 --header 'X-Canary: 1' \
  --route-log data/logs/route_decisions_canary.jsonl \
  --out tests/gold/shadow_canary_turns.jsonl \
  --report tests/gold/shadow_canary_report.json
```

4. 回滚阈值（任一满足即回滚，改回 100% 8000）：

| 指标 | 阈值 |
| --- | --- |
| HTTP 5xx 率 | > 2% |
| `evidence_status=missing` 占比 | > 15% |
| 路由 p95 延迟 | > 基线 1.5 倍 |
| 用户点踩率 | 明显高于基线 |

5. 验收完成后**必须恢复 Nginx 原始配置并 reload**，确认 `/health` 和 `/api/stats` 正常。
   原始配置备份在改动前生成，恢复后对比 `nginx -t` 输出。

## 6. 一键顺序

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

python tests/eval/generate_gold.py
python tests/eval/validate_gold.py
python tests/eval/eval_routing.py --workers 4
python tests/eval/calibrate_thresholds.py
ROUTE_THRESHOLDS_PATH=tests/gold/calibrated_thresholds.json \
  python tests/eval/eval_routing.py --workers 4 \
  --out tests/gold/eval_report_calibrated.json \
  --predictions tests/gold/predictions_calibrated.jsonl
```
