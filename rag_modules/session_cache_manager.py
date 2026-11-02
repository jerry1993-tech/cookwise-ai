"""
会话缓存管理模块。

两种后端：
- 传了 redis_client：会话状态、上下文、语义缓存全部写 Redis，多个后端副本共享；
- 没传（单测/本地无 Redis）：退回进程内 dict，保证老测试和本地启动不被阻塞。

语义缓存仍然是“每会话独立 + 话题隔离”，只是存储位置从内存换成了 Redis。
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SessionCacheManager:
    """会话级语义缓存、上下文和结构化状态。"""

    def __init__(self, embedding_model=None, redis_client=None, ttl_seconds: int = 86400):
        self.embedding_model = embedding_model

        # session_id -> {query: {response, timestamp, topic_id}}
        self.session_caches = {}
        # session_id -> {query: embedding}
        self.session_embeddings = {}
        # session_id -> [messages]
        self.session_contexts = {}
        # session_id -> 结构化状态（改写/路由/缓存使用）
        self.session_states = {}

        # 生产环境由 main.py 注入；None 表示当前进程内模式（仅本地/测试）
        self.redis = redis_client
        self.session_ttl = int(ttl_seconds)
        self._prefix = "cw:session"

        # 缓存配置
        self.cache_threshold = 0.75  # 语义相似度阈值
        self.max_session_cache_size = 50  # 每个会话最大缓存条目数
        self.max_context_length = 10  # 每个会话保留的最大上下文消息数

    # ------------------------------------------------------------------
    # Redis key / 序列化小工具
    # ------------------------------------------------------------------

    def _key(self, session_id: str, suffix: str) -> str:
        return f"{self._prefix}:{session_id}:{suffix}"

    def _cache_key(self, session_id: str) -> str:
        return self._key(session_id, "cache")

    def _embedding_key(self, session_id: str) -> str:
        return self._key(session_id, "embedding")

    def _order_key(self, session_id: str) -> str:
        return self._key(session_id, "cache_order")

    def _context_key(self, session_id: str) -> str:
        return self._key(session_id, "context")

    def _state_key(self, session_id: str) -> str:
        return self._key(session_id, "state")

    def _redis_session_keys(self, session_id: str) -> List[str]:
        """一个会话的全部 Redis key；清理/刷新 TTL 时统一遍历。"""
        return [
            self._cache_key(session_id),
            self._embedding_key(session_id),
            self._order_key(session_id),
            self._context_key(session_id),
            self._state_key(session_id),
        ]

    @staticmethod
    def _text(value) -> str:
        """Redis 配 decode_responses=False，key 可能返回 bytes。"""
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _embedding_from_bytes(raw) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.float32)

    def _expire_session_keys(self, session_id: str) -> None:
        """一次刷新该会话所有 key 的 TTL；调用方已经在 Redis 可用分支里。"""
        pipe = self.redis.pipeline()
        for key in self._redis_session_keys(session_id):
            pipe.expire(key, self.session_ttl)
        pipe.execute()

    def _touch_sessions(self, session_id: str) -> None:
        """维护活跃会话集合；顺带清掉超过 TTL 的旧记录，统计接口不用扫全库。"""
        now = time.time()
        pipe = self.redis.pipeline()
        pipe.zadd(f"{self._prefix}s", {session_id: now})
        pipe.zremrangebyscore(f"{self._prefix}s", 0, now - self.session_ttl)
        pipe.execute()

    # ------------------------------------------------------------------
    # 语义缓存
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个向量的余弦相似度。"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def _pick_best(
        self,
        query_embedding: np.ndarray,
        cache_items: Dict[str, Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        current_topic: str,
    ) -> Optional[str]:
        """在候选缓存里找相似度最高且话题一致的一条。"""
        best_similarity = 0
        best_response = None

        for cached_query, cached_data in cache_items.items():
            if not isinstance(cached_data, dict):
                continue
            cached_topic = str(cached_data.get("topic_id") or "")
            # 话题切换后，不命中旧话题的缓存
            if current_topic and cached_topic and current_topic != cached_topic:
                continue

            cached_embedding = embeddings.get(cached_query)
            if cached_embedding is None:
                continue

            similarity = self._calculate_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity and similarity >= self.cache_threshold:
                best_similarity = similarity
                best_response = cached_data.get("response")

        if best_response:
            logger.info("🎯 会话缓存命中! 相似度: %.3f", best_similarity)
        return best_response

    def check_semantic_cache(self, query: str, session_id: str = None) -> Optional[str]:
        """检查会话级语义缓存中是否有相似查询。"""
        if not session_id:
            return None

        if self.redis is not None:
            try:
                # 状态读不到时宁可这次不命中缓存，也不要拿空 topic 去匹配别的会话内容
                current_topic = str(
                    self._get_redis_state(session_id).get("active_topic") or ""
                )
            except Exception as exc:
                logger.warning("读取 Redis 会话状态失败，跳过语义缓存: %s", exc)
                return None

            try:
                cached = self.redis.hgetall(self._cache_key(session_id))
                if not cached:
                    return None

                cache_items = {}
                for raw_query, raw_data in cached.items():
                    try:
                        cache_items[self._text(raw_query)] = json.loads(raw_data)
                    except (TypeError, ValueError):
                        continue

                raw_embeddings = self.redis.hgetall(self._embedding_key(session_id))
                embeddings = {}
                for raw_query, raw_vector in raw_embeddings.items():
                    key = self._text(raw_query)
                    if key not in cache_items:
                        continue
                    try:
                        embeddings[key] = self._embedding_from_bytes(raw_vector)
                    except ValueError:
                        continue

                query_embedding = self.embedding_model.embed_documents([query])[0]
                return self._pick_best(query_embedding, cache_items, embeddings, current_topic)
            except Exception as exc:
                logger.warning("读取 Redis 语义缓存失败: %s", exc)
                return None

        # 进程内模式（本地/测试）
        if session_id not in self.session_caches:
            return None
        session_cache = self.session_caches[session_id]
        session_embeddings = self.session_embeddings[session_id]
        if not session_cache:
            return None

        current_topic = str(self.get_state(session_id).get("active_topic") or "")

        try:
            query_embedding = self.embedding_model.embed_documents([query])[0]
        except Exception as exc:
            logger.warning("查询向量计算失败: %s", exc)
            return None

        return self._pick_best(query_embedding, session_cache, session_embeddings, current_topic)

    def add_to_semantic_cache(self, query: str, response: str, session_id: str = None):
        """把查询-答案对写入会话级语义缓存。"""
        if not session_id:
            return

        try:
            query_embedding = self.embedding_model.embed_documents([query])[0]
        except Exception as exc:
            logger.warning("添加语义缓存失败（向量计算）: %s", exc)
            return

        if self.redis is not None:
            try:
                topic_id = str(self._get_redis_state(session_id).get("active_topic") or "")
                payload = json.dumps(
                    {
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "topic_id": topic_id,
                    },
                    ensure_ascii=False,
                )
                cache_key = self._cache_key(session_id)
                embedding_key = self._embedding_key(session_id)
                order_key = self._order_key(session_id)

                pipe = self.redis.pipeline()
                pipe.hset(cache_key, query, payload)
                pipe.hset(
                    embedding_key,
                    query,
                    np.asarray(query_embedding, dtype=np.float32).tobytes(),
                )
                pipe.zadd(order_key, {query: time.time()})
                pipe.execute()

                # 超过会话上限时，按写入时间淘汰最旧的一条
                if int(self.redis.hlen(cache_key)) > self.max_session_cache_size:
                    oldest = self.redis.zrange(order_key, 0, 0)
                    if oldest:
                        old_query = self._text(oldest[0])
                        self.redis.hdel(cache_key, old_query)
                        self.redis.hdel(embedding_key, old_query)
                        self.redis.zrem(order_key, old_query)

                self._expire_session_keys(session_id)
                self._touch_sessions(session_id)
                logger.info("📝 已写入 Redis 会话缓存 %s", session_id)
                return
            except Exception as exc:
                logger.warning("写入 Redis 语义缓存失败: %s", exc)
                return

        # 进程内模式（本地/测试）
        try:
            if session_id not in self.session_caches:
                self.session_caches[session_id] = {}
                self.session_embeddings[session_id] = {}

            session_cache = self.session_caches[session_id]
            session_embeddings = self.session_embeddings[session_id]

            if len(session_cache) >= self.max_session_cache_size:
                oldest_key = next(iter(session_cache))
                del session_cache[oldest_key]
                del session_embeddings[oldest_key]

            session_cache[query] = {
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "topic_id": str(self.session_states.get(session_id, {}).get("active_topic") or ""),
            }
            session_embeddings[query] = query_embedding
            logger.info("📝 已添加到会话缓存 %s, 当前大小: %d", session_id, len(session_cache))
        except Exception as exc:
            logger.warning("添加到语义缓存失败: %s", exc)

    # ------------------------------------------------------------------
    # 上下文
    # ------------------------------------------------------------------

    def add_to_context(self, session_id: str, query: str, response: str):
        """添加一轮对话到上下文历史。"""
        if not session_id:
            return

        message = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }

        if self.redis is not None:
            try:
                context_key = self._context_key(session_id)
                pipe = self.redis.pipeline()
                pipe.rpush(context_key, json.dumps(message, ensure_ascii=False))
                pipe.ltrim(context_key, -self.max_context_length, -1)
                # 一次刷新该会话所有 key 的 TTL，避免状态和上下文不同时过期
                for key in self._redis_session_keys(session_id):
                    pipe.expire(key, self.session_ttl)
                pipe.execute()
                self._touch_sessions(session_id)
                return
            except Exception as exc:
                logger.warning("写入 Redis 上下文失败: %s", exc)
                return

        # 进程内模式（本地/测试）
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = []
        context = self.session_contexts[session_id]
        context.append(message)
        if len(context) > self.max_context_length:
            context.pop(0)

    def get_context_for_query(self, session_id: str, current_query: str) -> str:
        """兼容旧调用：返回“可用于检索的独立 query”，不再拼接 AI 答案。"""
        try:
            from .query_routing.domain import DOMAIN

            state = self.get_state(session_id)
            active_recipe = str(state.get("active_recipe") or "")
            if (
                active_recipe
                and active_recipe not in current_query
                and any(term in current_query for term in DOMAIN.reference_terms)
            ):
                return f"{active_recipe} {current_query}"
            return current_query
        except Exception as exc:
            logger.warning("上下文获取失败: %s", exc)
            return current_query

    # ------------------------------------------------------------------
    # 结构化会话状态
    # ------------------------------------------------------------------

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        """一个会话的初始状态。"""
        return {
            "active_recipe": "",
            "active_topic": "",
            "constraints": {},
            "last_route_key": "",
            "last_intent": "",
            "recent_user_queries": [],
            "updated_at": "",
        }

    @classmethod
    def _normalize_state(cls, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """补齐默认字段，并保证两个可变字段是新的 list/dict。"""
        merged = cls._default_state()
        merged.update(state or {})
        merged["constraints"] = dict(merged.get("constraints") or {})
        merged["recent_user_queries"] = list(merged.get("recent_user_queries") or [])
        return merged

    def _get_redis_state(self, session_id: str) -> Dict[str, Any]:
        """读取 Redis 状态；连接/解析失败会抛异常，让写路径自己决定跳过。"""
        raw = self.redis.get(self._state_key(session_id))
        if not raw:
            return self._default_state()
        return self._normalize_state(json.loads(raw))

    def get_state(self, session_id: str) -> Dict[str, Any]:
        """读取会话状态；没有就返回空结构。"""
        if self.redis is not None:
            try:
                return self._get_redis_state(session_id)
            except Exception as exc:
                logger.warning("读取 Redis 会话状态失败: %s", exc)
                return self._default_state()

        state = self.session_states.get(session_id)
        return self._normalize_state(state) if state else self._default_state()

    def get_history_for_rewrite(self, session_id: str, max_turns: int = 2) -> str:
        """只返回最近几轮用户原话 + 已知约束，供理解层做指代消解。"""
        state = self.get_state(session_id)
        if not state:
            return ""

        lines = [
            f"用户: {query}"
            for query in (state.get("recent_user_queries") or [])[-max_turns:]
        ]
        constraints = state.get("constraints") or {}
        if constraints:
            lines.append(f"已知约束: {constraints}")
        return "\n".join(lines)

    def _apply_state_update(self, state: Dict[str, Any], query: str, analysis, response: str = "") -> Dict[str, Any]:
        """把一轮分析结果合并进状态；Redis 和进程内两种模式共用这段逻辑。"""
        understanding = getattr(analysis, "understanding", None) or {}
        intents = understanding.get("intents") or []

        recipe_names = []
        for item in intents:
            for entity in item.get("entities") or []:
                if entity.get("type") == "recipe" and entity.get("name"):
                    recipe_names.append(entity["name"])

        topic_shift = bool(getattr(analysis, "topic_shift", False))
        route_key = str(getattr(analysis, "route_key", "") or "")
        topic_id = str(getattr(analysis, "topic_id", "") or "")

        if route_key == "out_of_scope":
            # 域外问题视为话题中断：清掉旧菜名，后续“它…”不能复用旧菜谱和缓存
            state["active_recipe"] = ""
            state["active_topic"] = "out_of_scope"
        elif topic_shift:
            # 切换话题时清掉旧菜名；如果新话题本身点了菜，就记新菜
            state["active_recipe"] = recipe_names[0] if recipe_names else ""
            state["active_topic"] = recipe_names[0] if recipe_names else ""
        else:
            if recipe_names:
                state["active_recipe"] = recipe_names[0]
            state["active_topic"] = topic_id or state.get("active_recipe") or route_key

        slots = getattr(analysis, "slots", None) or {}
        if isinstance(slots, dict):
            from .query_routing.domain import DOMAIN

            for key in DOMAIN.persisted_slots:
                if slots.get(key) not in (None, "", []):
                    state["constraints"][key] = slots[key]

        recent = state.setdefault("recent_user_queries", [])
        recent.append(query)
        if len(recent) > 4:
            del recent[:-4]

        state["last_route_key"] = route_key
        state["last_intent"] = str(getattr(analysis, "intent", "") or "")
        state["updated_at"] = datetime.now().isoformat()
        return state

    def _redis_update_state(
        self, session_id: str, query: str, analysis, response: str
    ) -> bool:
        """用 WATCH/MULTI 更新会话状态；并发改同一会话时自动重试，避免后写覆盖先写。"""
        from redis.exceptions import WatchError

        state_key = self._state_key(session_id)
        for _ in range(3):
            with self.redis.pipeline() as pipe:
                try:
                    pipe.watch(state_key)
                    raw = pipe.get(state_key)
                    state = (
                        self._normalize_state(json.loads(raw))
                        if raw
                        else self._default_state()
                    )
                    state = self._apply_state_update(state, query, analysis, response)

                    pipe.multi()
                    pipe.set(
                        state_key,
                        json.dumps(state, ensure_ascii=False, default=str),
                        ex=self.session_ttl,
                    )
                    pipe.execute()
                    self._expire_session_keys(session_id)
                    self._touch_sessions(session_id)
                    return True
                except WatchError:
                    # 另一个请求刚改过状态，重新读一遍再合并
                    continue
        logger.warning("会话状态并发更新冲突，跳过本轮: %s", session_id)
        return False

    def update_state_after_turn(
        self,
        session_id: str,
        query: str,
        analysis,
        response: str = "",
    ) -> None:
        """一轮问答结束后更新结构化状态；不把 AI 答案写进状态。"""
        if not session_id:
            return

        if self.redis is not None:
            try:
                self._redis_update_state(session_id, query, analysis, response)
            except Exception as exc:
                logger.warning("写入 Redis 会话状态失败: %s", exc)
            return

        # 进程内模式（本地/测试）
        state = self.session_states.setdefault(session_id, self._default_state())
        self._apply_state_update(state, query, analysis, response)

    # ------------------------------------------------------------------
    # 统计与清理
    # ------------------------------------------------------------------

    def get_session_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        if self.redis is not None:
            total_cached = 0
            total_contexts = 0
            try:
                # 统计前先清掉超过 TTL 的会话记录，避免数字只增不减
                self.redis.zremrangebyscore(
                    f"{self._prefix}s", 0, time.time() - self.session_ttl
                )
                for key in self.redis.scan_iter(f"{self._prefix}:*:cache", count=200):
                    total_cached += int(self.redis.hlen(key))
                for key in self.redis.scan_iter(f"{self._prefix}:*:context", count=200):
                    total_contexts += int(self.redis.llen(key))
                total_sessions = int(self.redis.zcard(f"{self._prefix}s"))
            except Exception as exc:
                logger.warning("读取 Redis 会话统计失败: %s", exc)
                total_sessions = 0
            return {
                "total_sessions": total_sessions,
                "total_cached_queries": total_cached,
                "total_contexts": total_contexts,
                "total_states": total_sessions,
                "cache_threshold": self.cache_threshold,
                "max_session_cache_size": self.max_session_cache_size,
                "max_context_length": self.max_context_length,
            }

        return {
            "total_sessions": len(self.session_caches),
            "total_cached_queries": sum(len(cache) for cache in self.session_caches.values()),
            "total_contexts": sum(len(context) for context in self.session_contexts.values()),
            "total_states": len(self.session_states),
            "cache_threshold": self.cache_threshold,
            "max_session_cache_size": self.max_session_cache_size,
            "max_context_length": self.max_context_length,
        }

    def clear_session_cache(self, session_id: str):
        """清除指定会话的缓存。"""
        if self.redis is not None:
            try:
                self.redis.delete(*self._redis_session_keys(session_id))
                self.redis.zrem(f"{self._prefix}s", session_id)
            except Exception as exc:
                logger.warning("清除 Redis 会话缓存失败: %s", exc)
            return

        if session_id in self.session_caches:
            del self.session_caches[session_id]
        if session_id in self.session_embeddings:
            del self.session_embeddings[session_id]
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
        if session_id in self.session_states:
            del self.session_states[session_id]
        logger.info("🗑️ 已清除会话 %s 的缓存", session_id)

    def clear_all_caches(self):
        """清除所有缓存。"""
        if self.redis is not None:
            try:
                keys = list(self.redis.scan_iter(f"{self._prefix}:*", count=500))
                if keys:
                    self.redis.delete(*keys)
                self.redis.delete(f"{self._prefix}s")
            except Exception as exc:
                logger.warning("清除 Redis 全部缓存失败: %s", exc)
            return

        self.session_caches.clear()
        self.session_embeddings.clear()
        self.session_contexts.clear()
        self.session_states.clear()
        logger.info("🗑️ 已清除所有会话缓存")
