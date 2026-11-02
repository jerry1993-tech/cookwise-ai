# 真实多轮 E2E

`multiturn_e2e.py` 通过运行中的 FastAPI/uvicorn `/api/chat` 验证多轮上下文、约束延续、换菜和主题切换。

## 前置

```bash
conda activate cookwise-ai
# 后端（默认 8000）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup python main.py >> /tmp/cookwise-backend.log 2>&1 &
```

Neo4j / Milvus 需要可用；路由决策默认写入 `data/logs/route_decisions.jsonl`。

## 运行

```bash
python tests/e2e/multiturn_e2e.py
```

可覆盖的变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `E2E_BASE_URL` | `http://localhost:8000` | 后端地址 |
| `E2E_ROUTE_LOG` | `data/logs/route_decisions.jsonl` | 路由决策 JSONL |
| `E2E_BACKEND_LOG` | `/tmp/cookwise-backend.log` | 后端 stdout 日志（JSONL 不可用时兜底） |
| `E2E_TIMEOUT` | `180` | 单轮请求超时（秒） |

脚本刻意不以 `test_` 开头，避免 pytest 在没有后端时误收集。CI 只跑离线单测，不跑本脚本。
