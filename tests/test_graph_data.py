"""
测试图数据准备模块：验证 load_graph_data / build_recipe_documents / chunk_documents
运行前请确保 Neo4j 服务已启动，且已导入菜谱数据。
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 将项目根目录加入模块搜索路径，支持从任意位置直接运行本脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_modules import GraphDataPreparationModule

# 读取 .env 里的连接配置
load_dotenv()


def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "all-in-rag")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    # 创建模块实例，会自动连接 Neo4j
    module = GraphDataPreparationModule(uri, user, password, database)

    # 第一步：加载图数据
    node_stats = module.load_graph_data()
    print("【图数据加载结果】")
    print(f"  菜谱节点数: {node_stats['recipes']}")
    print(f"  食材节点数: {node_stats['ingredients']}")
    print(f"  烹饪步骤节点数: {node_stats['cooking_steps']}")

    # 第二步：构建菜谱文档
    documents = module.build_recipe_documents()
    print(f"\n【构建菜谱文档】共 {len(documents)} 篇")

    # 第三步：分块
    chunks = module.chunk_documents(chunk_size=500, chunk_overlap=50)
    print(f"【文档分块】共 {len(chunks)} 块")

    # 第四步：打印统计信息
    print("\n【统计信息】")
    stats = module.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 关闭连接
    module.close()


if __name__ == "__main__":
    main()