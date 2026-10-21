"""
测试图RAG检索模块
前置：Neo4j 已启动并导入菜谱数据
.env 已配置 OPENAI_API_KEY
"""


import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 先加载 .env 里的环境变量，再 import config（config.py 在导入时就会读环境变量）
load_dotenv()

# 将项目根目录加入模块搜索路径，支持从任意位置直接运行本脚本
# 注意：必须在 import config 之前执行，否则 config 模块找不到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from config import GraphRAGConfig
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval


def build_llm_client():
    """构造一个 OpenAI 兼容的 LLM 客户端（和第5讲生成模块里用的是同一种写法）"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请先在 .env 里配置 OPENAI_API_KEY（见第2讲）")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


def main():
    # 1. 准备配置与 LLM 客户端
    config = GraphRAGConfig()
    llm_client = build_llm_client()

    # 2. 实例化并初始化图RAG检索模块（连 Neo4j + 构建图索引）
    module = GraphRAGRetrieval(config=config, llm_client=llm_client)
    module.initialize()

    query = "川菜跟鲁菜的区别是什么？"

    # 3. 测试一：理解图查询意图
    print("=" * 40)
    print("测试一：understand_graph_query")
    graph_query = module.understand_graph_query(query)
    print("识别出的查询类型(query_type):", graph_query.query_type.value)
    print("核心实体(source_entities):", graph_query.source_entities)
    print("目标实体(target_entities):", graph_query.target_entities)
    print("最大遍历深度(max_depth):", graph_query.max_depth)

    # 4. 测试二：图RAG检索
    print("=" * 40)
    print("测试二：graph_rag_search")
    results = module.graph_rag_search(query, top_k=5)
    print("检索结果条数:", len(results))
    for i, doc in enumerate(results):
        print(f"--- 结果 {i + 1} ---")
        print("内容(page_content):", doc.page_content)
        print("元数据(metadata):", doc.metadata)

    # 5. 关闭连接
    module.close()


if __name__ == "__main__":
    main()