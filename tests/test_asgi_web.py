"""FastAPI ASGI shell tests with a fake synchronous rag_system."""

from fastapi.testclient import TestClient

from rag_modules.web_service_handler import create_app


class FakeCache:
    def check_semantic_cache(self, query, session_id):
        return None

    def add_to_context(self, session_id, query, response):
        pass

    def add_to_semantic_cache(self, query, response, session_id):
        pass

    def get_session_stats(self):
        return {"total_sessions": 0}


class FakeRouter:
    def route_query(self, query, top_k):
        return [], {}

    def get_route_statistics(self):
        return {"total_queries": 1}


class FakeGeneration:
    def generate_adaptive_answer(self, query, documents):
        return "测试回答"

    def generate_adaptive_answer_stream(self, query, documents):
        yield "测试"
        yield "回答"


class FakeRecipeManager:
    def get_random_recipes_with_images(self, limit=3):
        return []

    def get_recipe_by_id(self, recipe_id):
        return {"id": recipe_id}


class FakeConfig:
    top_k = 5
    chat_rate_limit_per_minute = 30
    rate_limit_per_minute = 60


class FakeRagSystem:
    def __init__(self):
        self.cache_manager = FakeCache()
        self.query_router = FakeRouter()
        self.generation_module = FakeGeneration()
        self.recipe_manager = FakeRecipeManager()
        self.config = FakeConfig()


def test_health_chat_and_stats():
    client = TestClient(create_app(FakeRagSystem()))

    assert client.get("/health").status_code == 200
    response = client.post("/api/chat", json={"message": "测试", "session_id": "s1"})
    assert response.status_code == 200
    assert response.json()["response"] == "测试回答"
    assert client.get("/api/stats").status_code == 200


def test_rate_limit_returns_429():
    class FakeLimiter:
        chat_limit = 30
        default_limit = 60

        def __init__(self):
            self.calls = 0

        def allow(self, key, limit):
            if not key.startswith("chat:"):
                return True, 0
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


def test_stream_and_recipe():
    client = TestClient(create_app(FakeRagSystem()))

    stream = client.post(
        "/api/chat_stream", json={"message": "测试", "session_id": "s2"}
    )
    assert stream.status_code == 200
    assert "测试" in stream.text and "[DONE]" in stream.text

    recipe = client.get("/api/recipes/r1")
    assert recipe.status_code == 200
    assert recipe.json()["data"]["id"] == "r1"
