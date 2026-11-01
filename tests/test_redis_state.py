"""
Redis 会话状态 / 语义缓存 / 限流的集成测试。

本机没启动 Redis 时自动 skip，方便离线环境继续跑其它测试。
运行前可先执行：docker compose up -d redis
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

redis = pytest.importorskip("redis")

from rag_modules.redis_store import RedisRateLimiter
from rag_modules.session_cache_manager import SessionCacheManager


@pytest.fixture()
def client():
    client = redis.Redis.from_url("redis://localhost:6379/15", decode_responses=False)
    try:
        client.ping()
    except redis.RedisError:
        pytest.skip("本机没有可用的 Redis")
    client.flushdb()
    yield client
    client.flushdb()


class FakeEmbedding:
    """同一个 query 返回同一个向量；不同 query 方向略有差异。"""

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(4, dtype=np.float32)
            vector[0] = 1.0
            vector[1] = 0.1 if "宫保" in text else 0.0
            vector[2] = 0.1 if "水煮" in text else 0.0
            vectors.append(vector)
        return vectors


def test_session_state_and_semantic_cache_use_redis(client):
    manager = SessionCacheManager(
        embedding_model=FakeEmbedding(),
        redis_client=client,
        ttl_seconds=60,
    )
    analysis = SimpleNamespace(
        understanding={"intents": [{"entities": [{"name": "宫保鸡丁", "type": "recipe"}]}]},
        topic_shift=False,
        route_key="recipe_ingredients",
        topic_id="宫保鸡丁",
        slots={"spicy": "不辣"},
        intent="recipe_detail",
    )

    manager.update_state_after_turn("s1", "宫保鸡丁要哪些食材", analysis, "ok")
    manager.add_to_context("s1", "宫保鸡丁要哪些食材", "鸡肉、花生")
    manager.add_to_semantic_cache("宫保鸡丁要哪些食材", "鸡肉、花生", "s1")

    # 状态和缓存必须真的落在 Redis，而不是只改了进程内 dict
    assert client.exists("cw:session:s1:state")
    assert manager.get_state("s1")["active_recipe"] == "宫保鸡丁"
    assert manager.check_semantic_cache("宫保鸡丁要哪些食材", "s1") == "鸡肉、花生"
    assert manager.get_session_stats()["total_cached_queries"] == 1

    manager.clear_session_cache("s1")
    assert not client.exists("cw:session:s1:state")


def test_concurrent_state_updates_do_not_lose_queries(client):
    """同一会话两轮并发更新时，WATCH/MULTI 要保证两轮 query 都留下。"""
    manager = SessionCacheManager(
        embedding_model=FakeEmbedding(), redis_client=client, ttl_seconds=60
    )
    barrier = threading.Barrier(2)

    def update(query):
        analysis = SimpleNamespace(
            understanding={"intents": []},
            topic_shift=False,
            route_key="chitchat",
            topic_id="",
            slots={},
            intent="chitchat",
        )
        barrier.wait(timeout=2)
        manager.update_state_after_turn("s2", query, analysis)

    threads = [threading.Thread(target=update, args=(q,)) for q in ("第一轮", "第二轮")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    recent = manager.get_state("s2")["recent_user_queries"]
    assert sorted(recent) == ["第一轮", "第二轮"]


def test_redis_rate_limiter_blocks_after_limit(client):
    limiter = RedisRateLimiter(client, default_limit=1, chat_limit=1)
    assert limiter.allow("chat:127.0.0.1", 1)[0] is True
    allowed, retry_after = limiter.allow("chat:127.0.0.1", 1)
    assert allowed is False
    assert retry_after >= 1
