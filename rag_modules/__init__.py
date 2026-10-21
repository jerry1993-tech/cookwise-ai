"""
cookwise-ai 图 RAG 模块包。

整体链路：
    图数据准备 → 图索引 → 向量索引 → 检索 → 智能路由 → 答案生成 → Web 服务

公开名称按需加载，`import rag_modules` 不会引入 neo4j / pymilvus / openai 等重型依赖。
"""

import importlib

__version__ = "0.3.0"

# 公开名称 -> (子模块名, 属性名)，注释即链路顺序
_LAZY = {
    # 图数据准备
    "GraphNode": ("graph_data_preparation", "GraphNode"),
    "GraphRelation": ("graph_data_preparation", "GraphRelation"),
    "GraphDataPreparationModule": ("graph_data_preparation", "GraphDataPreparationModule"),
    # 图索引
    "EntityKeyValue": ("graph_indexing", "EntityKeyValue"),
    "RelationKeyValue": ("graph_indexing", "RelationKeyValue"),
    "GraphIndexingModule": ("graph_indexing", "GraphIndexingModule"),
    # 向量索引
    "MilvusIndexConstructionModule": ("milvus_index_construction", "MilvusIndexConstructionModule"),
    # 检索
    "RetrievalResult": ("hybrid_retrieval", "RetrievalResult"),
    "HybridRetrievalModule": ("hybrid_retrieval", "HybridRetrievalModule"),
    "QueryType": ("graph_rag_retrieval", "QueryType"),
    "GraphQuery": ("graph_rag_retrieval", "GraphQuery"),
    "GraphRAGRetrieval": ("graph_rag_retrieval", "GraphRAGRetrieval"),
    # 智能路由
    "SearchStrategy": ("intelligent_query_router", "SearchStrategy"),
    "QueryAnalysis": ("intelligent_query_router", "QueryAnalysis"),
    "IntelligentQueryRouter": ("intelligent_query_router", "IntelligentQueryRouter"),
    # 答案生成
    "GenerationIntegrationModule": ("generation_integration", "GenerationIntegrationModule"),
    # 会话、推荐与 Web 服务
    "SessionCacheManager": ("session_cache_manager", "SessionCacheManager"),
    "RecipeRecommendationManager": ("recipe_recommendation", "RecipeRecommendationManager"),
    "WebServiceHandler": ("web_service_handler", "WebServiceHandler"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    """按需导入子模块，避免 `import rag_modules` 时加载全部重型依赖。"""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__():
    return sorted(__all__)
