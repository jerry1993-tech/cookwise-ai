"""
混合检索的离线单测：RRF 融合 + BM25 中文分词。

不依赖 Neo4j / Milvus / LLM，直接跑：
    python -m pytest -q tests/test_hybrid_fusion.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag_modules.hybrid_retrieval import HybridRetrievalModule

fuse = HybridRetrievalModule._rrf_merge


def make_doc(node_id, content="正文"):
    return Document(page_content=content, metadata={"node_id": node_id})


def test_multi_source_hit_ranks_first():
    """三路都出现的文档，必须排第一。"""
    docs = fuse([
        ("dual_level", [make_doc("A"), make_doc("B")]),
        ("vector", [make_doc("B"), make_doc("C")]),
        ("bm25", [make_doc("B"), make_doc("D")]),
    ], top_k=3)
    assert docs[0].metadata["node_id"] == "B"


def test_rank_sum_beats_single_win():
    """A 只在一路拿第一，B 在三路都第二，B 的总分应该更高。"""
    docs = fuse([
        ("dual_level", [make_doc("A"), make_doc("B")]),
        ("vector", [make_doc("C"), make_doc("B")]),
        ("bm25", [make_doc("D"), make_doc("B")]),
    ], top_k=2)
    assert [doc.metadata["node_id"] for doc in docs] == ["B", "A"]


def test_same_source_chunks_counted_once():
    """同一路里一道菜的多个 chunk，只按最好名次算分一次。"""
    docs = fuse([
        ("bm25", [make_doc("A"), make_doc("A", "第二段"), make_doc("A", "第三段")]),
    ], top_k=1)
    assert docs[0].metadata["rrf_score"] == 1 / 61
    assert docs[0].metadata["rrf_chunk_hits"] == {"bm25": 3}


def test_canonical_doc_uses_best_rank():
    """同一道菜在向量路排第 1、在图路排第 2，正文取向量路那条。"""
    graph_doc = make_doc("A", "图路正文")
    vector_doc = make_doc("A", "向量路正文")
    docs = fuse([
        ("dual_level", [make_doc("X"), graph_doc]),
        ("vector", [vector_doc]),
    ], top_k=1)
    assert docs[0].page_content == "向量路正文"
    assert docs[0].metadata["rrf_sources"] == ["dual_level", "vector"]


def test_rrf_does_not_mutate_input():
    """融合结果用新 Document，不改动上游传入的对象。"""
    origin = make_doc("A")
    docs = fuse([("vector", [origin])], top_k=1)
    assert "rrf_score" not in origin.metadata
    assert docs[0].metadata["rrf_score"] > 0


def test_empty_and_top_k():
    assert fuse([("dual_level", []), ("vector", []), ("bm25", [])], top_k=5) == []
    docs = [make_doc(str(index)) for index in range(10)]
    assert len(fuse([("vector", docs)], top_k=3)) == 3


def test_tokenize_drops_stopwords():
    """BM25 建索引靠它：中文要切词，且去掉「需要/什么」这类虚词。"""
    tokens = HybridRetrievalModule._tokenize_chinese("红烧肉需要什么食材")
    assert "红烧肉" in tokens
    assert "需要" not in tokens and "什么" not in tokens


def test_bm25_search_without_index_returns_empty():
    """BM25 没建起来时安静返回空榜，不影响其它链路。"""
    module = HybridRetrievalModule(config=None, milvus_module=None, data_module=None, llm_client=None)
    assert module.bm25_search("宫保鸡丁", top_k=5) == []


def test_attach_parent_documents_replaces_top_n():
    """父文档回填：前 top_n 条换成整篇菜谱，顺序和条数不变。"""
    parent = Document(page_content="整篇父菜谱", metadata={"node_id": "A"})
    data_module = SimpleNamespace(documents=[parent])
    module = HybridRetrievalModule(
        config=None, milvus_module=None, data_module=data_module, llm_client=None
    )
    module._parent_doc_map = module._build_parent_doc_map()

    chunks = [Document(page_content="chunk", metadata={"node_id": "A"})]
    result = module._attach_parent_documents(chunks)
    assert result[0].page_content == "整篇父菜谱"
    assert chunks[0].page_content == "chunk"  # 原对象没被改


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"PASS {name}")
