"""
完整重建 Milvus 向量索引：Neo4j → 菜谱文档 → 分块 → 向量化 → 落库。

用法（项目根目录或任意位置均可）：
    python scripts/rebuild_index.py

注意：
    - 会先删除并重建 cooking_knowledge 集合，建议先停止 main.py 服务。
    - 需要 Neo4j 与 Milvus 已启动，且 .env 中连接配置正确。
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config import DEFAULT_CONFIG
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.milvus_index_construction import MilvusIndexConstructionModule


def main() -> int:
    os.chdir(ROOT)  # 确保相对路径（如 data/）以项目根目录为基准
    cfg = DEFAULT_CONFIG

    print("[1/4] 从 Neo4j 加载图数据...", flush=True)
    t0 = time.time()
    data = GraphDataPreparationModule(
        uri=cfg.neo4j_uri,
        user=cfg.neo4j_user,
        password=cfg.neo4j_password,
        database=cfg.neo4j_database,
    )
    stats = data.load_graph_data()
    print(f"      {stats}  ({time.time() - t0:.1f}s)", flush=True)

    print("[2/4] 构建菜谱文档并分块...", flush=True)
    t0 = time.time()
    data.build_recipe_documents()
    chunks = data.chunk_documents(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    print(f"      chunks={len(chunks)}  ({time.time() - t0:.1f}s)", flush=True)
    data.close()

    print("[3/4] 连接 Milvus、加载嵌入模型并重建索引...", flush=True)
    t0 = time.time()
    milvus = MilvusIndexConstructionModule(
        host=cfg.milvus_host,
        port=cfg.milvus_port,
        collection_name=cfg.milvus_collection_name,
        dimension=cfg.milvus_dimension,
        model_name=cfg.embedding_model,
    )
    ok = milvus.build_vector_index(chunks)
    print(f"      build_vector_index={ok}  ({time.time() - t0:.1f}s)", flush=True)

    print("[4/4] 校验...", flush=True)
    time.sleep(3)  # 等 Milvus 统计刷新
    print("      stats:", milvus.get_collection_stats(), flush=True)
    print(
        "      真实行数:",
        milvus.client.query(
            cfg.milvus_collection_name, filter="", output_fields=["count(*)"]
        ),
        flush=True,
    )
    results = milvus.similarity_search("川菜有什么特色菜", k=3)
    for i, r in enumerate(results, 1):
        print(
            f"      {i}. score={r['score']:.4f} "
            f"recipe={r['metadata'].get('recipe_name')} "
            f"text={r['text'][:30]}...",
            flush=True,
        )

    milvus.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
