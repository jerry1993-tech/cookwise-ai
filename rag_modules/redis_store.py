"""
Redis 相关的两个小工具：

1. create_redis_client：建连接并 ping 一次；
2. RedisRateLimiter：固定窗口限流，计数放在 Redis，多个后端副本共享配额。

这里只放“连接”和“限流”，会话/语义缓存仍由 SessionCacheManager 负责。
"""

import logging
import time
from typing import Optional, Tuple

import redis

logger = logging.getLogger(__name__)


def create_redis_client(url: Optional[str] = None, required: bool = False):
    """创建 Redis 连接。

    required=True 时连不上直接抛错（生产环境让启动失败，避免悄悄退回单机状态）；
    required=False 时返回 None，调用方可以退化到进程内实现（本地开发/单测）。
    """
    url = url or "redis://localhost:6379/0"
    client = redis.Redis.from_url(
        url,
        # 会话向量直接用 numpy bytes 存，不让 redis-py 把二进制当文本解码
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )
    try:
        client.ping()
        logger.info("Redis 连接成功: %s", url)
        return client
    except redis.RedisError as exc:
        client.close()  # 连接没建起来，顺手释放连接池
        if required:
            raise RuntimeError(f"Redis 不可用: {exc}") from exc
        logger.warning("Redis 不可用，降级为进程内状态（仅限本地/测试）: %s", exc)
        return None


class RedisRateLimiter:
    """固定窗口限流器。

    每个 key 每分钟一个计数桶：INCR 后第一次设置过期时间，
    超过 limit 就返回 429 需要的 Retry-After。
    """

    def __init__(
        self,
        redis_client,
        default_limit: int = 60,
        chat_limit: int = 30,
        window_seconds: int = 60,
    ):
        self.redis = redis_client
        self.default_limit = int(default_limit)
        self.chat_limit = int(chat_limit)
        self.window_seconds = int(window_seconds)

    def allow(self, key: str, limit: Optional[int] = None) -> Tuple[bool, int]:
        """返回 (是否放行, 建议重试秒数)。

        Redis 故障时选择放行：限流器不能成为新的单点故障；
        生产环境 Redis 是 required，连接问题会在启动时暴露。
        """
        if self.redis is None:
            return True, 0

        limit = int(limit or self.default_limit)
        bucket = f"cw:rate:{key}:{int(time.time() // self.window_seconds)}"
        try:
            count = self.redis.incr(bucket)
            if count == 1:
                # 只在第一次计数时设置过期，避免每次请求都刷新 TTL
                self.redis.expire(bucket, self.window_seconds + 1)
            if count > limit:
                ttl = self.redis.ttl(bucket)
                retry_after = (
                    int(ttl) if isinstance(ttl, int) and ttl > 0 else self.window_seconds
                )
                return False, max(1, retry_after)
        except redis.RedisError as exc:
            logger.warning("Redis 限流失败，本次请求放行: %s", exc)
            return True, 0
        return True, 0
