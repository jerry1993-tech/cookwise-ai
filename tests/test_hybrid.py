"""
测试混合检索模块

运行前请确保：
1. Neo4j 与 Milvus 服务已启动
2. 完成图数据准备、第4讲完成Milvus索引构建
3. .env 中已配置 OPENAI_API_KEY / OPENAI_BASE_URL（LLM 用于提取关键词）
"""

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# 必须在导入 config 之前加载 .env（见第2讲）
load_dotenv()

# 将项目根目录加入模块搜索路径，支持从任意位置直接运行本脚本
# 注意：必须在 import config 之前执行，否则 config 模块找不到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.milvus_index_construction import MilvusIndexConstructionModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.hybrid_retrieval import HybridRetrievalModule

logging.basicConfig(level=logging.INFO)


def main():
    config = DEFAULT_CONFIG

    # 1. 图数据准备模块（第3讲）
    print("初始化数据准备模块...")
    data_module = GraphDataPreparationModule(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database
    )
    data_module.load_graph_data()
    data_module.build_recipe_documents()
    chunks = data_module.chunk_documents(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )

    # 2. Milvus向量索引模块（第4讲）
    print("初始化Milvus向量索引...")
    milvus_module = MilvusIndexConstructionModule(
        host=config.milvus_host,
        port=config.milvus_port,
        collection_name=config.milvus_collection_name,
        dimension=config.milvus_dimension,
        model_name=config.embedding_model
    )
    if milvus_module.has_collection():
        milvus_module.load_collection()
    else:
        milvus_module.build_vector_index(chunks)

    # 3. 生成模块（第5讲）—— 提供 llm_client 用于关键词提取
    print("初始化生成模块...")
    generation_module = GenerationIntegrationModule(
        model_name=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    # 4. 混合检索模块（本讲）
    print("初始化混合检索模块...")
    hybrid = HybridRetrievalModule(
        config=config,
        milvus_module=milvus_module,
        data_module=data_module,
        llm_client=generation_module.client
    )
    hybrid.initialize(chunks)

    # 5. 执行混合检索
    query = "川菜有什么特色菜"
    docs = hybrid.hybrid_search(query, top_k=5)

    print(f"\n查询：{query}")
    print(f"返回文档数量：{len(docs)}")
    for doc in docs:
        name = doc.metadata.get("recipe_name", "未知菜品")
        method = doc.metadata.get("search_method", "?")
        score = doc.metadata.get("final_score", 0.0)
        print(f"  - {name} | 方法={method} | final_score={score:.4f}")

    hybrid.close()


if __name__ == "__main__":
    main()