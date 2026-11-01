"""
Web服务处理模块
负责处理Web API和静态文件服务

生产入口是 FastAPI/uvicorn 的 create_app；
业务层保持同步，路由、生成、缓存、检索都在 Starlette 线程池里执行，
避免阻塞事件循环。
"""

import inspect
import ipaddress
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

# 只信任本机回环和 Docker 默认网桥；公网/普通局域网直连不能伪造 X-Real-IP
_TRUSTED_PROXY_NETWORKS = tuple(
    ipaddress.ip_network(item)
    for item in ("127.0.0.0/8", "::1/128", "172.16.0.0/12")
)


def _sse(payload) -> str:
    """SSE 数据帧：data: {json}\\n\\n。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class WebServiceHandler:
    """Web 服务处理器：提供路由、生成、会话、限流、反馈等同步辅助方法。"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self._feedback_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 路由 / 生成 / 会话状态的小工具
    # ------------------------------------------------------------------

    def _route_with_session(self, query: str, session_id: str):
        """调用路由器；兼容不接受 session_id 的旧路由实现/测试替身。"""
        route = self.rag_system.query_router.route_query
        top_k = getattr(self.rag_system.config, "top_k", 5)

        try:
            params = inspect.signature(route).parameters
        except (TypeError, ValueError):
            return route(query, top_k)

        kwargs = {}
        if "session_id" in params:
            kwargs["session_id"] = session_id
            get_state = getattr(self.rag_system.cache_manager, "get_state", None)
            if get_state and "session_state" in params:
                kwargs["session_state"] = get_state(session_id)
        return route(query, top_k, **kwargs)

    def _respond(self, documents, analysis, query: str) -> str:
        """按 response_type 选择：澄清 / 直接回答 / 证据回答。"""
        response_type = getattr(analysis, "response_type", "answer")
        if response_type == "clarify":
            return getattr(analysis, "clarify_question", "") or "请再说明一下你的问题。"

        question = getattr(analysis, "standalone_query", "") or query
        needs_retrieval = getattr(analysis, "needs_retrieval", True)
        # 证据缺失时不把兜底文档交给生成层，避免编造数值
        if response_type == "no_data":
            return self._call_generation("generate_adaptive_answer", question, [], analysis)
        if response_type == "refuse" or not needs_retrieval:
            direct = getattr(self.rag_system.generation_module, "generate_direct_answer", None)
            if direct is not None:
                return direct(question, analysis)
            return self._call_generation("generate_adaptive_answer", question, [], analysis)

        return self._call_generation("generate_adaptive_answer", question, documents, analysis)

    def _call_generation(self, method_name: str, question: str, documents, analysis) -> str:
        """兼容旧生成模块：没有 analysis 参数时按旧签名调用。"""
        method = getattr(self.rag_system.generation_module, method_name, None)
        if method is None:
            return ""
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return method(question, documents)
        if "analysis" in params:
            return method(question, documents, analysis=analysis)
        return method(question, documents)

    def _call_generation_stream(self, method_name: str, question: str, documents, analysis):
        """流式生成的兼容包装。"""
        method = getattr(self.rag_system.generation_module, method_name, None)
        if method is None:
            return
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):
            yield from method(question, documents)
            return
        if "analysis" in params:
            yield from method(question, documents, analysis=analysis)
        else:
            yield from method(question, documents)

    def _stream_response(self, documents, analysis, query: str):
        """按 response_type 选择流式回答路径。"""
        response_type = getattr(analysis, "response_type", "answer")
        question = getattr(analysis, "standalone_query", "") or query

        if response_type == "clarify":
            text = getattr(analysis, "clarify_question", "") or "请再说明一下你的问题。"
            for index in range(0, len(text), 3):
                yield text[index:index + 3]
            return

        needs_retrieval = getattr(analysis, "needs_retrieval", True)
        if response_type == "no_data":
            yield from self._call_generation_stream(
                "generate_adaptive_answer_stream", question, [], analysis
            )
            return
        if response_type == "refuse" or not needs_retrieval:
            direct_stream = getattr(self.rag_system.generation_module, "generate_direct_answer_stream", None)
            if direct_stream is not None:
                yield from direct_stream(question, analysis)
            else:
                yield from self._call_generation_stream(
                    "generate_adaptive_answer_stream", question, [], analysis
                )
            return

        yield from self._call_generation_stream(
            "generate_adaptive_answer_stream", question, documents, analysis
        )

    def _update_session_state(self, session_id: str, query: str, analysis, response: str) -> None:
        """更新结构化会话状态；测试替身没有这个方法时安全跳过。"""
        updater = getattr(self.rag_system.cache_manager, "update_state_after_turn", None)
        if updater is None:
            return
        try:
            updater(session_id, query, analysis, response)
        except TypeError:
            updater(session_id, query, analysis)
        except Exception as exc:
            logger.warning(f"会话状态更新失败: {exc}")

    @staticmethod
    def _is_trusted_proxy(address: str) -> bool:
        """只有本机回环或 Docker 网桥来的请求才相信转发头。"""
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(ip in network for network in _TRUSTED_PROXY_NETWORKS)

    @classmethod
    def _client_key_from(cls, remote: str, headers) -> str:
        """限流 key 用客户端 IP；直连公网时忽略可伪造的 X-Real-IP。"""
        forwarded = headers.get("X-Real-IP") or ""
        if not forwarded:
            forwarded = headers.get("X-Forwarded-For", "").split(",")[0].strip()

        if forwarded and cls._is_trusted_proxy(remote):
            return forwarded
        return remote or "unknown"

    def _chat_limit(self) -> int:
        limiter = getattr(self.rag_system, "rate_limiter", None)
        if limiter is not None:
            return int(getattr(limiter, "chat_limit", 30))
        return int(getattr(self.rag_system.config, "chat_rate_limit_per_minute", 30))

    def _default_limit(self) -> int:
        limiter = getattr(self.rag_system, "rate_limiter", None)
        if limiter is not None:
            return int(getattr(limiter, "default_limit", 60))
        return int(getattr(self.rag_system.config, "rate_limit_per_minute", 60))

    # ------------------------------------------------------------------
    # 聊天核心流程
    # ------------------------------------------------------------------

    def _chat_once(self, query: str, session_id: str):
        """普通聊天核心流程：语义缓存 → 路由 → 生成 → 更新状态/缓存。"""
        # 语义缓存命中：直接返回，但不要跳过会话上下文记录
        cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
        if cached_response:
            self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
            return {
                "response": cached_response,
                "query": query,
                "session_id": session_id,
                "timestamp": str(datetime.now()),
                "from_cache": True
            }

        # 路由：把原始 query 和结构化状态交给路由器，不在 Web 层拼接历史答案
        documents, analysis = self._route_with_session(query, session_id)
        response = self._respond(documents, analysis, query)

        # 先更新状态（确定本轮 topic_id），再写语义缓存；澄清/拒识/无数据不缓存
        self._update_session_state(session_id, query, analysis, response)
        if getattr(analysis, "response_type", "answer") not in ("no_data", "clarify", "refuse"):
            self.rag_system.cache_manager.add_to_semantic_cache(query, response, session_id)
        self.rag_system.cache_manager.add_to_context(session_id, query, response)

        return {
            "response": response,
            "query": query,
            "timestamp": str(datetime.now())
        }

    def _stream_events(self, query: str, session_id: str):
        """流式聊天核心流程，逐个产出 SSE 文本帧。"""
        try:
            cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
            if cached_response:
                self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
                for index in range(0, len(cached_response), 3):
                    yield _sse({
                        "chunk": cached_response[index:index + 3],
                        "from_cache": True
                    })
                    time.sleep(0.02)
                yield "data: [DONE]\n\n"
                return

            documents, analysis = self._route_with_session(query, session_id)

            full_response = ""
            for chunk in self._stream_response(documents, analysis, query):
                full_response += chunk
                yield _sse({"chunk": chunk})

            self._update_session_state(session_id, query, analysis, full_response)
            if full_response and getattr(analysis, "response_type", "answer") not in ("no_data", "clarify", "refuse"):
                self.rag_system.cache_manager.add_to_semantic_cache(query, full_response, session_id)
            self.rag_system.cache_manager.add_to_context(session_id, query, full_response)

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream API错误: {e}")
            error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
            yield _sse({"chunk": error_msg})
            yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # 反馈落盘与统计
    # ------------------------------------------------------------------

    @staticmethod
    def _feedback_path():
        """反馈日志 path；和路由日志一样用 JSONL，不额外起数据库。"""
        return os.getenv("FEEDBACK_LOG_PATH", "data/logs/feedback.jsonl")

    def _build_feedback_record(self, data):
        """校验反馈并构造待落盘记录；返回 (record, error)。"""
        rating = str(data.get("rating") or "").strip().lower()
        correction = str(data.get("correction") or "").strip()
        if rating not in ("like", "dislike") and not correction:
            return None, "rating 必须是 like/dislike，或提供 correction"

        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": str(data.get("session_id") or ""),
            "query": str(data.get("query") or ""),
            "answer": str(data.get("answer") or ""),
            "rating": rating,
            "correction": correction,
            "route_key": str(data.get("route_key") or ""),
        }
        return record, None

    def _append_feedback(self, record):
        """把反馈追加到 JSONL；失败时抛出 OSError。"""
        path = self._feedback_path()
        with self._feedback_lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_feedback_counts(self):
        """统计 JSONL 里的反馈数；文件不存在时返回 0。"""
        counts = {"like": 0, "dislike": 0, "correction": 0, "total": 0}
        try:
            with open(self._feedback_path(), encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    counts["total"] += 1
                    rating = str(item.get("rating") or "")
                    if rating in ("like", "dislike"):
                        counts[rating] += 1
                    if item.get("correction"):
                        counts["correction"] += 1
        except FileNotFoundError:
            pass
        return counts


def create_app(rag_system) -> FastAPI:
    """创建 FastAPI 应用；HTTP 层用 ASGI，业务逻辑复用 WebServiceHandler。"""
    handler = WebServiceHandler(rag_system)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Retry-After"],
    )

    def client_key(request: Request) -> str:
        """取限流 key；只有内网代理来的请求才采信转发头。"""
        remote = request.client.host if request.client else ""
        return handler._client_key_from(remote, request.headers)

    async def rate_limited(request: Request, scope: str, limit: int):
        """返回 None 表示放行；返回 JSONResponse 表示已经触发 429。"""
        limiter = getattr(rag_system, "rate_limiter", None)
        if limiter is None:
            return None

        # Redis 限流是同步调用，放进线程池，别卡住事件循环
        allowed, retry_after = await run_in_threadpool(
            limiter.allow, f"{scope}:{client_key(request)}", limit
        )
        if allowed:
            return None
        return JSONResponse(
            {"error": "请求过于频繁，请稍后再试"},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    @app.get("/health")
    def health():
        """健康检查接口；配置了 Redis 时顺带确认它还能 ping 通。"""
        redis_client = getattr(rag_system.cache_manager, "redis", None)
        if redis_client is not None:
            try:
                redis_client.ping()
            except Exception as exc:
                logger.warning("健康检查 Redis 失败: %s", exc)
                return JSONResponse(
                    {"status": "unhealthy", "service": "RAG System", "redis": "down"},
                    status_code=503,
                )
        return {
            "status": "healthy",
            "timestamp": str(datetime.now()),
            "service": "RAG System",
        }

    @app.post("/api/chat")
    async def chat(request: Request):
        """聊天API - 普通响应"""
        limited = await rate_limited(request, "chat", handler._chat_limit())
        if limited is not None:
            return limited
        try:
            data = await request.json()
            query = data.get("message", "")
            session_id = data.get("session_id", "")
            if not query:
                return JSONResponse({"error": "消息不能为空"}, status_code=400)
            # 同步业务放到线程池，避免阻塞事件循环
            return await run_in_threadpool(handler._chat_once, query, session_id)
        except Exception as exc:
            logger.error("Chat API 错误: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/chat_stream")
    async def chat_stream(request: Request):
        """聊天API - 流式响应"""
        limited = await rate_limited(request, "chat", handler._chat_limit())
        if limited is not None:
            return limited
        try:
            data = await request.json()
            query = data.get("message", "")
            session_id = data.get("session_id", "")
            if not query:
                return JSONResponse({"error": "消息不能为空"}, status_code=400)
            # 同步生成器交给 Starlette 在线程池里迭代
            return StreamingResponse(
                handler._stream_events(query, session_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as exc:
            logger.error("Stream Chat API 错误: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/recipes/recommendations")
    async def recommendations(request: Request):
        """获取菜谱推荐"""
        limited = await rate_limited(request, "recommend", handler._default_limit())
        if limited is not None:
            return limited
        try:
            await request.json()
        except Exception:
            pass
        try:
            # preferences 字段保留给后续个性化，当前仍返回随机推荐
            recipes = await run_in_threadpool(
                rag_system.recipe_manager.get_random_recipes_with_images, limit=3
            )
        except Exception as exc:
            logger.error("推荐API错误: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)
        return {"success": True, "data": recipes, "message": "推荐获取成功"}

    @app.get("/api/recipes/{recipe_id}")
    async def recipe_detail(recipe_id: str, request: Request):
        """获取菜谱详情"""
        limited = await rate_limited(request, "recipe_detail", handler._default_limit())
        if limited is not None:
            return limited
        try:
            recipe = await run_in_threadpool(
                rag_system.recipe_manager.get_recipe_by_id, recipe_id
            )
        except Exception as exc:
            logger.error("菜谱详情API错误: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)
        if recipe:
            return {"success": True, "data": recipe}
        return JSONResponse({"error": "菜谱不存在"}, status_code=404)

    @app.get("/api/stats")
    async def stats(request: Request):
        """获取系统统计信息"""
        limited = await rate_limited(request, "stats", handler._default_limit())
        if limited is not None:
            return limited
        try:
            return {
                "cache_stats": await run_in_threadpool(
                    rag_system.cache_manager.get_session_stats
                ),
                "route_stats": rag_system.query_router.get_route_statistics(),
                "system_info": {"timestamp": str(datetime.now()), "status": "running"},
            }
        except Exception as exc:
            logger.error("统计API错误: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/feedback")
    async def feedback(request: Request):
        """点踩/纠错入口，落 JSONL，供离线分析"""
        limited = await rate_limited(request, "feedback", handler._default_limit())
        if limited is not None:
            return limited
        try:
            data = await request.json()
        except Exception:
            data = {}
        record, error = handler._build_feedback_record(data)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        try:
            await run_in_threadpool(handler._append_feedback, record)
        except OSError as exc:
            logger.warning("反馈落盘失败: %s", exc)
            return JSONResponse({"error": "反馈暂存失败"}, status_code=500)
        return {"status": "ok"}

    @app.get("/api/feedback/stats")
    async def feedback_stats(request: Request):
        """反馈计数"""
        limited = await rate_limited(request, "feedback_stats", handler._default_limit())
        if limited is not None:
            return limited
        try:
            return await run_in_threadpool(handler._read_feedback_counts)
        except OSError as exc:
            logger.warning("读取反馈失败: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    def frontend_path(*parts) -> Path:
        """前端文件路径：frontend/dist。"""
        return Path(os.getcwd()) / "frontend" / "dist" / Path(*parts)

    @app.get("/")
    def index():
        """提供主页"""
        path = frontend_path("index.html")
        if path.exists():
            return FileResponse(path)
        return JSONResponse({"error": "Not Found"}, status_code=404)

    @app.get("/{full_path:path}")
    def static_file(full_path: str):
        """提供静态文件服务；找不到时回退 index.html，支持 SPA 路由。"""
        # 安全检查，防止路径遍历攻击
        if ".." in full_path or full_path.startswith("/"):
            return PlainTextResponse("Forbidden", status_code=403)

        path = frontend_path(full_path)
        if path.is_file():
            return FileResponse(path)

        # 如果文件不存在，返回index.html（用于SPA路由）
        index_path = frontend_path("index.html")
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"error": "Not Found"}, status_code=404)

    return app
