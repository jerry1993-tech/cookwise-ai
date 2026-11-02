"""
真实多轮 E2E：通过运行中的 FastAPI/uvicorn /api/chat 验证上下文、约束和主题切换。

前置：
- 后端已启动（默认 http://localhost:8000）
- Neo4j / Milvus 可用
- 路由决策日志已开启（默认 data/logs/route_decisions.jsonl）

用法：
    conda activate cookwise-ai
    python tests/e2e/multiturn_e2e.py
    E2E_BASE_URL=http://localhost:8001 python tests/e2e/multiturn_e2e.py

说明：
- 脚本按 session_id 从 JSONL 决策日志读取本轮 route_key / evidence_status；
- 日志不可用时退回解析后端 stdout 日志；
- 纯脚本，不以 test_ 开头，避免 pytest 在无后端环境误收集。
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROUTE_RE = re.compile(
    r"查询理解完成: intent=(\S+) route_key=(\S+) engine=(\S+) needs_retrieval=(\S+)"
)
COUNT_RE = re.compile(r"路由完成，返回 (\d+) 个结果")


class RouteReader:
    """优先读 JSONL 决策日志，读不到再解析后端 stdout。"""

    def __init__(self, route_log: Path, backend_log: Path):
        self.route_log = route_log
        self.backend_log = backend_log
        self.route_offset = 0
        self.backend_offset = 0

    @staticmethod
    def _size(path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    @staticmethod
    def _read_since(path: Path, offset: int) -> str:
        if not path.exists():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            return handle.read()

    def start(self) -> None:
        self.route_offset = self._size(self.route_log)
        self.backend_offset = self._size(self.backend_log)

    def finish(self, session_id: str, query: str) -> dict:
        """返回本轮路由信息；两种日志都没有时返回空字段。"""
        route = self._from_jsonl(session_id, query)
        if route:
            return route
        return self._from_backend_log()

    def _from_jsonl(self, session_id: str, query: str) -> dict:
        text = self._read_since(self.route_log, self.route_offset)
        self.route_offset = self._size(self.route_log)

        matched = None
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if item.get("session_id") == session_id and item.get("query") == query:
                matched = item
        if not matched:
            return {}
        return {
            "intent": matched.get("intent", "?"),
            "route_key": matched.get("route_key", "?"),
            "engine": matched.get("engine", "?"),
            "needs_retrieval": str(matched.get("needs_retrieval", "?")),
            "docs": int(matched.get("doc_count") or 0),
            "evidence_status": matched.get("evidence_status", ""),
            "confidence": matched.get("confidence"),
        }

    def _from_backend_log(self) -> dict:
        text = self._read_since(self.backend_log, self.backend_offset)
        self.backend_offset = self._size(self.backend_log)
        routes = ROUTE_RE.findall(text)
        counts = COUNT_RE.findall(text)
        route = routes[-1] if routes else ("?", "?", "?", "?")
        return {
            "intent": route[0],
            "route_key": route[1],
            "engine": route[2],
            "needs_retrieval": route[3],
            "docs": int(counts[-1]) if counts else 0,
            "evidence_status": "",
            "confidence": None,
        }


class E2ERunner:
    def __init__(self, base_url: str, route_log: Path, backend_log: Path, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.reader = RouteReader(route_log, backend_log)
        self.timeout = timeout
        self.passed = []
        self.failed = []

    def ask(self, session_id: str, query: str) -> dict:
        self.reader.start()
        started = time.time()
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"message": query, "session_id": session_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        time.sleep(0.3)  # 等后端把 JSONL 刷完
        route = self.reader.finish(session_id, query)
        return {
            "query": query,
            "answer": body.get("response", ""),
            "elapsed": round(time.time() - started, 1),
            **route,
        }

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed.append(name)
            print(f"PASS: {name}")
        else:
            self.failed.append((name, detail))
            print(f"FAIL: {name} -> {detail}")

    @staticmethod
    def show(turn: dict) -> None:
        answer = turn["answer"].replace("\n", " ")
        print(
            f"\n[{turn['query']}]\n"
            f"  route={turn['route_key']} intent={turn['intent']} "
            f"engine={turn['engine']} needs_retrieval={turn['needs_retrieval']} "
            f"docs={turn['docs']} evidence={turn.get('evidence_status') or '-'} "
            f"confidence={turn.get('confidence')} {turn['elapsed']}s\n"
            f"  answer={answer[:300]}"
        )

    def run_recipe_session(self) -> None:
        print("\n=== 场景 A：同一道菜的连续追问 + 换菜 + 主题切换 ===")
        sid = f"e2e-mt-recipe-{int(time.time())}"

        t1 = self.ask(sid, "做宫保鸡丁需要准备哪些食材和调料？")
        self.show(t1)
        self.check("A1 宫保鸡丁食材路由", t1["route_key"] == "recipe_ingredients", t1["route_key"])
        self.check(
            "A1 宫保鸡丁食材覆盖",
            all(word in t1["answer"] for word in ("鸡", "花生", "干辣椒", "花椒")),
            t1["answer"][:200],
        )
        self.check("A1 未混入其他菜食材", "豆腐" not in t1["answer"], t1["answer"][:200])

        t2 = self.ask(sid, "那要多少盐？")
        self.show(t2)
        self.check("A2 省略追问仍绑定宫保鸡丁", t2["route_key"] == "recipe_ingredients", t2["route_key"])
        self.check("A2 回答提到盐", "盐" in t2["answer"], t2["answer"][:200])
        self.check("A2 未串到水煮鱼", "水煮鱼" not in t2["answer"], t2["answer"][:200])

        t3 = self.ask(sid, "要煮多久？")
        self.show(t3)
        self.check("A3 时间追问路由", t3["route_key"] == "recipe_steps_time", t3["route_key"])
        self.check(
            "A3 回答包含时间",
            any(word in t3["answer"] for word in ("分钟", "秒", "小时")),
            t3["answer"][:200],
        )

        t4 = self.ask(sid, "水煮鱼需要哪些食材？")
        self.show(t4)
        self.check("A4 换菜后路由正确", t4["route_key"] == "recipe_ingredients", t4["route_key"])
        self.check(
            "A4 水煮鱼食材覆盖",
            "巴沙鱼" in t4["answer"] and ("蔬菜" in t4["answer"] or "豆芽" in t4["answer"]),
            t4["answer"][:250],
        )
        self.check("A4 未串入宫保鸡丁食材", "花生米" not in t4["answer"], t4["answer"][:250])

        t5 = self.ask(sid, "它需要哪些食材？")
        self.show(t5)
        self.check("A5 指代绑定当前菜（水煮鱼）", "巴沙鱼" in t5["answer"], t5["answer"][:250])
        self.check(
            "A5 未回退到上一道菜（宫保鸡丁）",
            "鸡胸" not in t5["answer"] and "花生米" not in t5["answer"],
            t5["answer"][:250],
        )

        t6 = self.ask(sid, "北京明天天气怎么样？")
        self.show(t6)
        self.check("A6 域外不检索", t6["route_key"] == "out_of_scope", t6["route_key"])
        self.check("A6 域外拒识", t6["needs_retrieval"] == "False", t6["needs_retrieval"])
        self.check("A6 未带出菜谱证据", "巴沙鱼" not in t6["answer"], t6["answer"][:200])

        t7 = self.ask(sid, "它需要哪些食材？")
        self.show(t7)
        # 主题切换后 active_recipe 应被清空；这里不能再答水煮鱼或宫保鸡丁
        self.check(
            "A7 主题切换后不复用旧菜谱",
            "巴沙鱼" not in t7["answer"] and "鸡胸" not in t7["answer"] and "花生米" not in t7["answer"],
            t7["answer"][:250],
        )
        self.check(
            "A7 主动澄清或要求补充菜名",
            any(word in t7["answer"] for word in ("哪道菜", "哪种食材", "补充", "说明")),
            t7["answer"][:250],
        )

    def run_constraint_session(self) -> None:
        print("\n=== 场景 B：约束延续（不吃辣 -> 推荐） ===")
        sid = f"e2e-mt-constraint-{int(time.time())}"

        t1 = self.ask(sid, "我不吃辣")
        self.show(t1)
        self.check("B1 纯约束不触发检索", t1["needs_retrieval"] == "False", t1["needs_retrieval"])

        t2 = self.ask(sid, "推荐几道适合我的菜")
        self.show(t2)
        self.check("B2 推荐路由", t2["route_key"] == "recommend_list", t2["route_key"])
        self.check(
            "B2 推荐遵守不吃辣约束",
            "水煮鱼" not in t2["answer"] and "宫保鸡丁" not in t2["answer"],
            t2["answer"][:300],
        )
        self.check("B2 有具体推荐", len(t2["answer"]) >= 40, t2["answer"][:100])

    def run_multi_intent_session(self) -> None:
        print("\n=== 场景 C：单句多意图 ===")
        sid = f"e2e-mt-multi-{int(time.time())}"

        t1 = self.ask(sid, "宫保鸡丁需要哪些食材？要煮多久？")
        self.show(t1)
        self.check(
            "C1 多意图包含食材与时间",
            "花生" in t1["answer"]
            and any(word in t1["answer"] for word in ("分钟", "秒")),
            t1["answer"][:350],
        )
        self.check("C1 证据状态落盘", bool(t1.get("evidence_status")), t1.get("evidence_status"))

    def run(self) -> int:
        health = requests.get(f"{self.base_url}/health", timeout=5)
        print("health:", health.status_code, health.text[:100])

        self.run_recipe_session()
        self.run_constraint_session()
        self.run_multi_intent_session()

        print("\n===== 汇总 =====")
        print(f"PASS {len(self.passed)} / FAIL {len(self.failed)}")
        for name, detail in self.failed:
            print(f"  FAIL: {name} -> {detail[:300]}")
        return 0 if not self.failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="CookWise 真实多轮 E2E")
    parser.add_argument(
        "--base-url",
        default=os.getenv("E2E_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--route-log",
        default=os.getenv("E2E_ROUTE_LOG", str(PROJECT_ROOT / "data/logs/route_decisions.jsonl")),
    )
    parser.add_argument(
        "--backend-log",
        default=os.getenv("E2E_BACKEND_LOG", "/tmp/cookwise-backend.log"),
    )
    parser.add_argument("--timeout", type=int, default=int(os.getenv("E2E_TIMEOUT", "180")))
    args = parser.parse_args()

    try:
        runner = E2ERunner(
            base_url=args.base_url,
            route_log=Path(args.route_log),
            backend_log=Path(args.backend_log),
            timeout=args.timeout,
        )
        return runner.run()
    except requests.RequestException as exc:
        print("后端未就绪:", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
