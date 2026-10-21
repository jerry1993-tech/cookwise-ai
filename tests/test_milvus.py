"""
测试Milvus索引构建模块
"""


import sys
from pathlib import Path
from langchain_core.documents import Document

# 将项目根目录加入模块搜索路径，支持从任意位置直接运行本脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_modules.milvus_index_construction import MilvusIndexConstructionModule


def make_doc(chunk_id: str, text: str, **metadata) -> Document:
    """造一个带 metadata 的假文档块，字段和第 3 讲产出一致"""
    meta = {
        "node_id": f"node_{chunk_id}",
        "recipe_name": "宫保鸡丁",
        "node_type": "Recipe",
        "category": "川菜",
        "cuisine_type": "川菜",
        "difficulty": 2,
        "doc_type": "recipe",
        "chunk_id": chunk_id,
        "parent_id": "recipe_gongbao",
    }
    meta.update(metadata)
    return Document(page_content=text, metadata=meta)


def main():
    # 1. 初始化模块：会先连 Milvus，再加载 BGE 嵌入模型（首次会下载模型）
    module = MilvusIndexConstructionModule()

    # 2. 造 4 个假文档块
    docs = [
        make_doc("chunk_1", "宫保鸡丁是一道经典的川菜，主要食材有鸡胸肉、花生米和干辣椒。"),
        make_doc("chunk_2", "宫保鸡丁的做法：先把鸡胸肉切丁腌制，再下锅爆炒，最后加入花生米翻炒。"),
        make_doc("chunk_3", "鱼香肉丝是川菜代表菜之一，口味酸甜微辣，主要食材有肉丝和木耳。"),
        make_doc("chunk_4", "麻婆豆腐是川菜名菜，用豆腐和牛肉末烧制，麻辣鲜香。"),
    ]

    # 3. 构建向量索引：会强制重建集合、批量插入、建 HNSW 索引、加载进内存
    ok = module.build_vector_index(docs)
    print(f"build_vector_index 结果: {ok}")

    # 4. 查看集合统计
    print("集合统计:", module.get_collection_stats())

    # 5. 相似度搜索
    query = "如何做宫保鸡丁？"
    results = module.similarity_search(query, k=3)
    print(f"\n查询: {query}")
    for r in results:
        print(f"  score={r['score']:.4f}  recipe={r['metadata']['recipe_name']}  text={r['text'][:20]}...")

    # 6. 带过滤条件的搜索：只搜难度为 2 的
    results2 = module.similarity_search(query, k=3, filters={"difficulty": 2})
    print(f"\n带 difficulty==2 过滤的结果数量: {len(results2)}")
    for r in results2:
        print(f"  score={r['score']:.4f}  difficulty={r['metadata']['difficulty']}")


if __name__ == "__main__":
    main()