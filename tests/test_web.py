"""
test_web.py - 验证 FastAPI/uvicorn Web 路由是否正常注册并响应
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from rag_modules.web_service_handler import WebServiceHandler, create_app


class FakeCacheManager:
    def check_semantic_cache(self, query, session_id):
        return None

    def get_context_for_query(self, session_id, query):
        return query

    def add_to_context(self, session_id, query, response):
        pass

    def add_to_semantic_cache(self, query, response, session_id):
        pass

    def get_session_stats(self):
        return {"sessions": 0}


class FakeQueryRouter:
    def route_query(self, query, top_k):
        return [], {}

    def get_route_statistics(self):
        return {"total": 0}


class FakeGenerationModule:
    def generate_adaptive_answer(self, query, documents):
        return "测试回答"

    def generate_adaptive_answer_stream(self, query, documents):
        yield "测试"
        yield "回答"


class FakeRecipeManager:
    def get_random_recipes_with_images(self, limit=3):
        return []

    def get_recipe_by_id(self, recipe_id):
        return {"id": recipe_id, "name": "测试菜谱"}


class FakeConfig:
    top_k = 5


class FakeRagSystem:
    def __init__(self):
        self.cache_manager = FakeCacheManager()
        self.query_router = FakeQueryRouter()
        self.generation_module = FakeGenerationModule()
        self.recipe_manager = FakeRecipeManager()
        self.config = FakeConfig()


def test_health_check():
    client = TestClient(create_app(FakeRagSystem()))
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_rate_limit_key_ignores_spoofed_forward_header():
    """公网直连时不能靠 X-Real-IP 换身份；只有内网代理来的请求才采信。"""
    assert WebServiceHandler._client_key_from(
        "8.8.8.8", {"X-Real-IP": "1.2.3.4"}
    ) == "8.8.8.8"

    assert WebServiceHandler._client_key_from(
        "172.18.0.2", {"X-Real-IP": "1.2.3.4"}
    ) == "1.2.3.4"


def test_chat_rate_limit_returns_429():
    """限流器说超限时，Web 层要在进入检索/生成之前返回 429。"""
    class FakeLimiter:
        def __init__(self):
            self.calls = 0

        def allow(self, key, limit):
            self.calls += 1
            return self.calls <= 1, 2

    system = FakeRagSystem()
    system.rate_limiter = FakeLimiter()
    client = TestClient(create_app(system))

    first = client.post("/api/chat", json={"message": "测试", "session_id": "s1"})
    second = client.post("/api/chat", json={"message": "测试", "session_id": "s1"})
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "2"


if __name__ == "__main__":
    test_health_check()
    test_rate_limit_key_ignores_spoofed_forward_header()
    test_chat_rate_limit_returns_429()
    print("test_web.py 通过：FastAPI /health 与限流返回正常")
