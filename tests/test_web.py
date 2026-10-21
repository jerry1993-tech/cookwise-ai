"""
test_web.py - 验证 web_service_handler 的 Flask 路由是否正常注册并响应
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_modules.web_service_handler import WebServiceHandler


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
    handler = WebServiceHandler(FakeRagSystem())
    app = handler.setup_flask_app()
    assert app is not None

    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


if __name__ == "__main__":
    test_health_check()
    print("test_web.py 通过：/health 返回 200 且 status=healthy")
