"""
路由决策落盘（JSONL）。

为什么用 JSONL 而不是数据库：
- 本项目是单机、低流量，追加写即可，不额外起服务、不占常驻内存；
- 影子回放 / 阈值校准直接逐行读取，不需要 SQL；
- 文件按大小滚动，避免无限增长。

每行就是一次 route_query 的决策快照，字段保持扁平、可离线聚合。
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RouteDecisionLogger:
    """把路由决策追加到 JSONL；写入失败只告警，不影响主链路。"""

    def __init__(
        self,
        path: Optional[str] = None,
        enabled: Optional[bool] = None,
        max_bytes: int = 20 * 1024 * 1024,
    ):
        self.path = Path(
            path or os.getenv("ROUTE_LOG_PATH", "data/logs/route_decisions.jsonl")
        )
        if enabled is None:
            enabled = os.getenv("ROUTE_LOG_ENABLED", "1").lower() not in (
                "0", "false", "no", "off"
            )
        self.enabled = enabled
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._warned = False

    def log(self, record: Dict[str, Any]) -> None:
        """追加一条记录；自动补时间戳，失败不抛异常。"""
        if not self.enabled:
            return

        payload = {"ts": datetime.now().isoformat(timespec="milliseconds"), **record}
        line = json.dumps(payload, ensure_ascii=False, default=str)

        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                if not self._warned:
                    logger.warning("路由决策落盘失败: %s", exc)
                    self._warned = True

    def _rotate_if_needed(self) -> None:
        """超过上限就改名归档；只保留一个滚动文件，避免占满磁盘。"""
        if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
            return
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        rotated = self.path.with_name(f"{self.path.stem}-{suffix}{self.path.suffix}")
        self.path.rename(rotated)
