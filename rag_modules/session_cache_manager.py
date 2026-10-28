"""
会话缓存管理模块
负责管理会话级语义缓存和上下文
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class SessionCacheManager:
    """
    会话级缓存管理器
    
    功能：
    1. 会话级语义缓存 - 每个聊天窗口独立缓存
    2. 上下文管理 - 维护对话历史
    3. 语义相似度匹配 - 智能缓存命中
    """
    
    def __init__(self, embedding_model=None):
        """初始化缓存管理器"""
        self.embedding_model = embedding_model
        
        # 🚀 会话级语义缓存系统 - 针对每个聊天窗口独立缓存
        self.session_caches = {}  # 按session_id分组的缓存：{session_id: {query: response}}
        self.session_embeddings = {}  # 按session_id分组的向量：{session_id: {query: embedding}}
        self.session_contexts = {}  # 按session_id分组的上下文：{session_id: [messages]}
        # 结构化会话状态：改写/路由/缓存使用；不把 AI 答案拼进检索 query
        self.session_states = {}
        
        # 缓存配置
        self.cache_threshold = 0.75  # 语义相似度阈值
        self.max_session_cache_size = 50  # 每个会话最大缓存条目数
        self.max_context_length = 10  # 每个会话保留的最大上下文消息数
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def check_semantic_cache(self, query: str, session_id: str = None) -> Optional[str]:
        """检查会话级语义缓存中是否有相似查询"""
        if not session_id or session_id not in self.session_caches:
            return None

        session_cache = self.session_caches[session_id]
        session_embeddings = self.session_embeddings[session_id]
        current_topic = str(self.session_states.get(session_id, {}).get("active_topic") or "")

        if not session_cache:
            return None

        try:
            # 计算查询向量
            query_embedding = self.embedding_model.embed_documents([query])[0]
        except Exception as e:
            logger.warning(f"查询向量计算失败: {e}")
            return None

        # 查找最相似的缓存查询
        best_similarity = 0
        best_response = None

        for cached_query, cached_data in session_cache.items():
            cached_topic = str(cached_data.get("topic_id") or "")
            # 话题切换后，不命中旧话题的缓存
            if current_topic and cached_topic and current_topic != cached_topic:
                continue

            cached_embedding = session_embeddings.get(cached_query)
            if cached_embedding is not None:
                similarity = self._calculate_similarity(query_embedding, cached_embedding)
                if similarity > best_similarity and similarity >= self.cache_threshold:
                    best_similarity = similarity
                    best_response = cached_data['response']

        if best_response:
            logger.info(f"🎯 会话缓存命中! Session: {session_id}, 相似度: {best_similarity:.3f}")
            return best_response

        return None

    def add_to_semantic_cache(self, query: str, response: str, session_id: str = None):
        """将查询-答案对添加到会话级语义缓存"""
        try:
            if not session_id:
                return

            # 初始化会话缓存
            if session_id not in self.session_caches:
                self.session_caches[session_id] = {}
                self.session_embeddings[session_id] = {}

            session_cache = self.session_caches[session_id]
            session_embeddings = self.session_embeddings[session_id]

            # 限制会话缓存大小
            if len(session_cache) >= self.max_session_cache_size:
                # 删除最旧的缓存项
                oldest_key = next(iter(session_cache))
                del session_cache[oldest_key]
                del session_embeddings[oldest_key]

            # 计算查询向量
            query_embedding = self.embedding_model.embed_documents([query])[0]

            # 添加到缓存（带上话题键，话题切换后旧缓存失效）
            session_cache[query] = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'topic_id': str(self.session_states.get(session_id, {}).get("active_topic") or ""),
            }
            session_embeddings[query] = query_embedding

            logger.info(f"📝 已添加到会话缓存 {session_id}, 当前大小: {len(session_cache)}")

        except Exception as e:
            logger.warning(f"添加到语义缓存失败: {e}")

    def add_to_context(self, session_id: str, query: str, response: str):
        """添加对话到上下文历史"""
        try:
            if not session_id:
                return

            # 初始化会话上下文
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = []

            context = self.session_contexts[session_id]

            # 添加新的对话
            context.append({
                'query': query,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })

            # 限制上下文长度
            if len(context) > self.max_context_length:
                context.pop(0)  # 删除最旧的对话

            logger.info(f"📝 已添加上下文到会话 {session_id}, 当前长度: {len(context)}")

        except Exception as e:
            logger.warning(f"添加上下文失败: {e}")

    def get_context_for_query(self, session_id: str, current_query: str) -> str:
        """
        兼容旧调用：返回“可用于检索的独立 query”，不再拼接 AI 答案。

        真正的理解/改写由路由器的理解层完成；这里只做一个最小改写，
        避免旧调用方继续把历史答案拼进检索向量。
        """
        try:
            from .query_routing.domain import DOMAIN

            state = self.session_states.get(session_id, {})
            active_recipe = str(state.get("active_recipe") or "")
            if (
                active_recipe
                and active_recipe not in current_query
                and any(term in current_query for term in DOMAIN.reference_terms)
            ):
                return f"{active_recipe} {current_query}"
            return current_query

        except Exception as e:
            logger.warning(f"上下文获取失败: {e}")
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

    def get_state(self, session_id: str) -> Dict[str, Any]:
        """读取会话状态；没有就返回空结构。"""
        state = self.session_states.get(session_id)
        if not state:
            return self._default_state()
        return {
            **state,
            "constraints": dict(state.get("constraints") or {}),
            "recent_user_queries": list(state.get("recent_user_queries") or []),
        }

    def get_history_for_rewrite(self, session_id: str, max_turns: int = 2) -> str:
        """只返回最近几轮用户原话 + 已知约束，供理解层做指代消解。"""
        state = self.session_states.get(session_id, {})
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

        state = self.session_states.setdefault(session_id, self._default_state())
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

    def get_session_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'total_sessions': len(self.session_caches),
            'total_cached_queries': sum(len(cache) for cache in self.session_caches.values()),
            'total_contexts': sum(len(context) for context in self.session_contexts.values()),
            'total_states': len(self.session_states),
            'cache_threshold': self.cache_threshold,
            'max_session_cache_size': self.max_session_cache_size,
            'max_context_length': self.max_context_length
        }

    def clear_session_cache(self, session_id: str):
        """清除指定会话的缓存"""
        if session_id in self.session_caches:
            del self.session_caches[session_id]
        if session_id in self.session_embeddings:
            del self.session_embeddings[session_id]
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
        if session_id in self.session_states:
            del self.session_states[session_id]
        logger.info(f"🗑️ 已清除会话 {session_id} 的缓存")

    def clear_all_caches(self):
        """清除所有缓存"""
        self.session_caches.clear()
        self.session_embeddings.clear()
        self.session_contexts.clear()
        self.session_states.clear()
        logger.info("🗑️ 已清除所有会话缓存")
