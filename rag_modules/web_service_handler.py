"""
Web服务处理模块
负责处理Web API和静态文件服务
"""

import inspect
import json
import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class WebServiceHandler:
    """初始化 Web 服务处理器"""
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.app = None  # Flask app will be set later
        self._feedback_lock = threading.Lock()

    def setup_flask_app(self):
        """创建 Flask 应用并挂载路由"""
        try:
            from flask import Flask
            from flask_cors import CORS

            self.app = Flask(__name__)
            CORS(self.app)  # 允许跨域请求

            # 设置路由
            self._set_routes()

            return self.app

        except ImportError as e:
            logger.error(f"Flask导入失败: {e}")
            return None

    def _set_routes(self):
        """设置所有API路由"""
        from flask import jsonify
        @self.app.route('/')
        def serve_index():
            """提供主页"""
            return self._serve_static_file('index.html')

        @self.app.route('/<path:filename>')
        def serve_static(filename):
            """提供静态文件服务"""
            return self._serve_static_file(filename)

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            return jsonify({
                "status": "healthy",
                "timestamp": str(datetime.now()),
                "service": "RAG System"
                })

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """聊天API - 普通响应"""
            return self._handle_chat_request()

        @self.app.route('/api/chat_stream', methods=['POST'])
        def chat_stream():
            """聊天API - 流式响应"""
            return self._handle_stream_request()

        @self.app.route('/api/recipes/recommendations', methods=['POST'])
        def get_recommendations():
            """获取菜谱推荐"""
            return self._handle_recommendations_request()

        @self.app.route('/api/recipes/<recipe_id>', methods=['GET'])
        def get_recipe_detail(recipe_id):
            """获取菜谱详情"""
            return self._handle_recipe_detail_request(recipe_id)

        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """获取系统统计信息"""
            return self._handle_stats_request()

        @self.app.route('/api/feedback', methods=['POST'])
        def submit_feedback():
            """点踩/纠错入口，落 JSONL，供离线分析"""
            return self._handle_feedback_request()

        @self.app.route('/api/feedback/stats', methods=['GET'])
        def get_feedback_stats():
            """反馈计数"""
            return self._handle_feedback_stats()

    def _serve_static_file(self, filename):
        """提供静态文件服务"""
        from flask import send_from_directory

        # 安全检查，防止路径遍历攻击
        if '..' in filename or filename.startswith('/'):
            return "Forbidden", 403

        # 前端文件路径
        frontend_path = os.path.join(os.getcwd(), 'frontend', 'dist')
        
        try:
            if filename == 'index.html' or filename == '':
                return send_from_directory(frontend_path, 'index.html')
            else:
                return send_from_directory(frontend_path, filename)
        except FileNotFoundError:
            # 如果文件不存在，返回index.html（用于SPA路由）
            return send_from_directory(frontend_path, 'index.html')

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

    def _handle_chat_request(self):
        """处理普通聊天请求"""
        from flask import request, jsonify

        try:
            data = request.get_json()
            query = data.get('message', '')
            session_id = data.get('session_id', '')

            if not query:
                return jsonify({"error": "消息不能为空"}), 400

            # 语义缓存命中：直接返回，但不要跳过会话上下文记录
            cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
            if cached_response:
                self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
                return jsonify({
                    "response": cached_response,
                    "query": query,
                    "session_id": session_id,
                    "timestamp": str(datetime.now()),
                    "from_cache": True
                })

            # 路由：把原始 query 和结构化状态交给路由器，不在 Web 层拼接历史答案
            documents, analysis = self._route_with_session(query, session_id)
            response = self._respond(documents, analysis, query)

            # 先更新状态（确定本轮 topic_id），再写语义缓存；澄清/拒识/无数据不缓存
            self._update_session_state(session_id, query, analysis, response)
            if getattr(analysis, "response_type", "answer") not in ("no_data", "clarify", "refuse"):
                self.rag_system.cache_manager.add_to_semantic_cache(query, response, session_id)
            self.rag_system.cache_manager.add_to_context(session_id, query, response)

            return jsonify({
                "response": response,
                "query": query,
                "timestamp": str(datetime.now())
            })

        except Exception as e:
            logger.error(f"Chat API 错误: {e}")
            return jsonify({"error": str(e)}), 500


    def _handle_stream_request(self):
        """处理流式聊天请求"""
        from flask import request, Response, jsonify

        try:
            data = request.get_json()
            query = data.get('message', '')
            session_id = data.get('session_id', '')

            if not query:
                return jsonify({"error": "消息不能为空"}), 400

            def generate():
                """生成器函数"""
                try:
                    cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
                    if cached_response:
                        self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
                        for i in range(0, len(cached_response), 3):
                            data_obj = {"chunk": cached_response[i:i + 3], "from_cache": True}
                            yield f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"
                            time.sleep(0.02)
                        yield "data: [DONE]\n\n"
                        return

                    documents, analysis = self._route_with_session(query, session_id)

                    full_response = ""
                    for chunk in self._stream_response(documents, analysis, query):
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"

                    self._update_session_state(session_id, query, analysis, full_response)
                    if full_response and getattr(analysis, "response_type", "answer") not in ("no_data", "clarify", "refuse"):
                        self.rag_system.cache_manager.add_to_semantic_cache(query, full_response, session_id)
                    self.rag_system.cache_manager.add_to_context(session_id, query, full_response)

                    yield "data: [DONE]\n\n"

                except Exception as e:
                    logger.error(f"Stream API错误: {e}")
                    error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
                    yield f"data: {json.dumps({'chunk': error_msg}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

            response = Response(generate(), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response

        except Exception as e:
            logger.error(f"Stream Chat API 错误: {e}")
            return jsonify({"error": str(e)}), 500


    def _handle_recommendations_request(self):
        """处理菜谱推荐请求"""
        from flask import request, jsonify
        
        try:
            data = request.get_json() or {}
            preferences = data.get('preferences', {})
            
            # 获取推荐菜谱
            recipes = self.rag_system.recipe_manager.get_random_recipes_with_images(limit=3)
            
            return jsonify({
                "success": True,
                "data": recipes,
                "message": "推荐获取成功"
            })
            
        except Exception as e:
            logger.error(f"推荐API错误: {e}")
            return jsonify({"error": str(e)}), 500

    
    def _handle_recipe_detail_request(self, recipe_id):
        """处理菜谱详情请求"""
        from flask import jsonify
        
        try:
            recipe = self.rag_system.recipe_manager.get_recipe_by_id(recipe_id)
            if recipe:
                return jsonify({
                    "success": True,
                    "data": recipe
                })
            else:
                return jsonify({"error": "菜谱不存在"}), 404
                
        except Exception as e:
            logger.error(f"菜谱详情API错误: {e}")
            return jsonify({"error": str(e)}), 500

    
    def _handle_stats_request(self):
        """处理统计信息请求"""
        from flask import jsonify
        
        try:
            # 获取系统统计信息
            stats = {
                "cache_stats": self.rag_system.cache_manager.get_session_stats(),
                "route_stats": self.rag_system.query_router.get_route_statistics(),
                "system_info": {
                    "timestamp": str(datetime.now()),
                    "status": "running"
                }
            }
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"统计API错误: {e}")
            return jsonify({"error": str(e)}), 500

    @staticmethod
    def _feedback_path():
        """反馈日志 path；和路由日志一样用 JSONL，不额外起数据库。"""
        return os.getenv("FEEDBACK_LOG_PATH", "data/logs/feedback.jsonl")

    def _handle_feedback_request(self):
        """接收点赞/点踩/纠错，只落盘，不阻塞聊天链路。"""
        from flask import request, jsonify

        data = request.get_json(silent=True) or {}
        rating = str(data.get("rating") or "").strip().lower()
        correction = str(data.get("correction") or "").strip()
        if rating not in ("like", "dislike") and not correction:
            return jsonify({"error": "rating 必须是 like/dislike，或提供 correction"}), 400

        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": str(data.get("session_id") or ""),
            "query": str(data.get("query") or ""),
            "answer": str(data.get("answer") or ""),
            "rating": rating,
            "correction": correction,
            "route_key": str(data.get("route_key") or ""),
        }
        path = self._feedback_path()
        try:
            with self._feedback_lock:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"反馈落盘失败: {exc}")
            return jsonify({"error": "反馈暂存失败"}), 500
        return jsonify({"status": "ok"})

    def _handle_feedback_stats(self):
        """统计 JSONL 里的反馈数；文件不存在时返回 0。"""
        from flask import jsonify

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
        except OSError as exc:
            logger.warning(f"读取反馈失败: {exc}")
            return jsonify({"error": str(exc)}), 500
        return jsonify(counts)
    