"""
智能查询路由器验证脚本
用假对象代替真实检索器和LLM，验证路由决策逻辑，无需启动Neo4j和Milvus
"""

import json
import sys
from pathlib import Path

from langchain_core.documents import Document

# 将项目根目录加入模块搜索路径，支持从任意位置直接运行本脚本
# 注意：必须在 import rag_modules 之前执行，否则找不到包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_modules.intelligent_query_router import IntelligentQueryRouter


# 1. 假配置对象：路由器只用到了 config.llm_model 这一个属性
class FakeConfig:
    def __init__(self):
        self.llm_model = "moonshot-v1-8k"


# 2. 假 LLM 客户端：仿照 OpenAI 风格接口，返回一段预设的 JSON 字符串
class FakeLLMClient:
    def __init__(self, content: str):
        self.chat = FakeChat(content)


class FakeChat:
    def __init__(self, content: str):
        self.completions = FakeCompletions(content)


class FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return FakeResponse(self._content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


# 3. 假检索器：返回固定的 Document 列表
class FakeTraditionalRetrieval:
    def hybrid_search(self, query, top_k=5):
        return [Document(page_content="传统检索结果：宫保鸡丁的详细做法")]


class FakeGraphRagRetrieval:
    def graph_rag_search(self, query, top_k=5):
        return [Document(page_content="图RAG检索结果：花椒与川菜风味的形成关系")]


def main():
    # 让假 LLM 返回一个「关系密集、需要图推理」的分析结果
    fake_result = {
        "query_complexity": 0.6,
        "relationship_intensity": 0.8,
        "reasoning_required": True,
        "entity_count": 3,
        "recommended_strategy": "graph_rag",
        "confidence": 0.85,
        "reasoning": "该查询涉及多个实体间的复杂关系，需要图结构推理"
    }
    fake_llm = FakeLLMClient(json.dumps(fake_result, ensure_ascii=False))

    router = IntelligentQueryRouter(
        traditional_retrieval=FakeTraditionalRetrieval(),
        graph_rag_retrieval=FakeGraphRagRetrieval(),
        llm_client=fake_llm,
        config=FakeConfig(),
    )

    query = "为什么川菜用花椒而不是胡椒？"

    # ① 单独调用 analyze_query
    analysis = router.analyze_query(query)
    print("=== analyze_query 分析结果 ===")
    print("查询复杂度:", analysis.query_complexity)
    print("关系密集度:", analysis.relationship_intensity)
    print("是否需要推理:", analysis.reasoning_required)
    print("实体数量:", analysis.entity_count)
    print("推荐策略:", analysis.recommended_strategy.value)
    print("置信度:", analysis.confidence)
    print("推荐理由:", analysis.reasoning)

    # ② 调用 explain_routing_decision 打印可读报告
    print()
    print(router.explain_routing_decision(query))

    # ③ 调用 route_query 走完整路由
    print("=== route_query 完整路由 ===")
    documents, analysis2 = router.route_query(query, top_k=4)
    print("返回文档数量:", len(documents))
    for doc in documents:
        print(" - 正文:", doc.page_content)
        print("   路由策略:", doc.metadata.get("route_strategy"))

    # ④ 打印路由统计
    print()
    print("=== 路由统计 ===")
    print(router.get_route_statistics())


if __name__ == "__main__":
    main()
