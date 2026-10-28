"""
混合检索模块（对外入口）
基于双层检索范式：实体级 + 主题级检索
结合 BM25（jieba 分词）、向量检索与图键值索引，使用 RRF 融合

具体实现拆在 hybrid_rag_retrieval_components/ 下，本文件只做装配与编排：
    tokenizer.py            中文分词 + 停用词
    bm25_index.py           BM25 建索引 / 检索
    graph_layer.py          图索引构建、Neo4j 邻居查询
    dual_level_retriever.py 双层检索（关键词 + 实体级 + 主题级）
    vector_retriever.py     向量检索 + 邻居增强
    fusion.py               RRF 融合
    parent_documents.py     父文档回填
"""

import concurrent.futures
import logging
from typing import Dict, List, Tuple

from neo4j import GraphDatabase
from langchain_core.documents import Document

from .graph_indexing import GraphIndexingModule
from .hybrid_rag_retrieval_components import (
    BM25Index,
    DualLevelRetriever,
    GraphIndexBuilder,
    RetrievalResult,
    RRF_K,
    VectorSearcher,
    attach_parent_documents,
    build_parent_doc_map,
    rrf_merge,
    tokenize_chinese,
)

logger = logging.getLogger(__name__)

__all__ = ["HybridRetrievalModule", "RetrievalResult"]


def _read_future(future: concurrent.futures.Future, default):
    """读取并行任务的结果；任务抛异常时返回 default，不连累其它链路。"""
    try:
        return future.result() or default
    except Exception as e:
        logger.error(f"检索任务失败: {e}")
        return default


class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级，基于图键值索引）
    2. BM25 关键词检索（jieba 分词 + 停用词过滤）
    3. 向量检索（Milvus）+ 一跳邻居扩展
    4. RRF (Reciprocal Rank Fusion) 融合三路结果
    """

    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None

        # 各组件先建好；依赖 Neo4j 的 driver 在 initialize() 连上后再注入
        self.bm25_index = BM25Index()
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self._graph_index = GraphIndexBuilder(None, self.graph_indexing, data_module)
        self._dual_level = DualLevelRetriever(
            None, self.graph_indexing, llm_client, config, data_module
        )
        self._vector = VectorSearcher(None, milvus_module)

    # 兼容原属性名：索引实际由 BM25Index 持有
    @property
    def bm25(self):
        return self.bm25_index.bm25

    @property
    def bm25_corpus_docs(self) -> List[Document]:
        return self.bm25_index.corpus_docs

    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化混合检索模块...")

        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )

        # 连接的 driver 注入给需要它的组件
        self._graph_index.driver = self.driver
        self._dual_level.driver = self.driver
        self._vector.driver = self.driver

        # 初始化 BM25（jieba 分词 + 中文停用词过滤）
        self.bm25_index.build(chunks)

        # 初始化图索引
        self._graph_index.build()

        # 初始化父文档映射，每个nodeid对应该chunk所属父文档的document
        self._parent_doc_map = self._build_parent_doc_map()
        logger.info(f"父文档映射构建完成，菜谱文档数: {len(self._parent_doc_map)}")

    @staticmethod
    def _tokenize_chinese(text: str) -> List[str]:
        """jieba 精确分词 + 停用词 / 空白 / 单字符过滤"""
        return tokenize_chinese(text)

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """提取查询关键词：实体级 + 主题级"""
        return self._dual_level.extract_query_keywords(query)

    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """实体级检索：专注于具体实体和关系"""
        return self._dual_level.entity_level_retrieval(entity_keywords, top_k)

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """主题级检索：专注于广泛主题和概念"""
        return self._dual_level.topic_level_retrieval(topic_keywords, top_k)

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """双层检索：结合实体级和主题级检索"""
        return self._dual_level.dual_level_retrieval(query, top_k)

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """增强的向量检索：结合图信息"""
        return self._vector.search(query, top_k)

    def bm25_search(self, query: str, top_k: int = 5) -> List[Document]:
        """BM25 检索：jieba 分词后查索引，按分数降序返回 top_k"""
        return self.bm25_index.search(query, top_k)

    @staticmethod
    def _rrf_merge(
        ranked_lists: List[Tuple[str, List[Document]]],
        top_k: int,
        k: int = RRF_K,
    ) -> List[Document]:
        """RRF 融合，实现见 hybrid_rag_retrieval_components/fusion.py"""
        return rrf_merge(ranked_lists, top_k, k)

    def _build_parent_doc_map(self) -> Dict[str, Document]:
        """{str(node_id): 整篇父菜谱 Document}"""
        return build_parent_doc_map(self.data_module)

    def _attach_parent_documents(self, docs: List[Document]) -> List[Document]:
        """父文档回填，实现见 hybrid_rag_retrieval_components/parent_documents.py"""
        if getattr(self, "_parent_doc_map", None) is None:
            self._parent_doc_map = self._build_parent_doc_map()
        return attach_parent_documents(
            docs,
            self._parent_doc_map,
            getattr(self.config, "parent_doc_top_n", 3),
            getattr(self.config, "parent_doc_max_chars", 4000),
        )

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        混合检索：三路召回（图键值双层 + 向量 + BM25）→ RRF 融合

        三路互不依赖，这里用线程池并行执行；融合结果与串行完全一致。
        """
        logger.info(f"开始混合检索（dual + vector + bm25, RRF k={RRF_K}）: {query}")

        # 每路给 RRF 留够候选空间，否则三路各自前 top_k 容易没交集，融合退化
        candidate_k = max(top_k * 2, 10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_dual = executor.submit(self.dual_level_retrieval, query, candidate_k)
            future_vector = executor.submit(self.vector_search_enhanced, query, candidate_k)
            future_bm25 = executor.submit(self.bm25_search, query, candidate_k)
        # with 退出时三路都已跑完，这里只负责取结果

        dual_docs = _read_future(future_dual, [])
        vector_docs = _read_future(future_vector, [])
        bm25_docs = _read_future(future_bm25, [])

        # 标记每路来源（dual_level 内部会写 search_type 但不一定写 search_method）
        for d in dual_docs:
            d.metadata.setdefault("search_method", "dual_level")
        for d in vector_docs:
            d.metadata["search_method"] = "vector"
        # bm25_search 内部已写 search_method=bm25

        final_docs = self._rrf_merge(
            ranked_lists=[
                ("dual_level", dual_docs),
                ("vector", vector_docs),
                ("bm25", bm25_docs),
            ],
            top_k=top_k,
        )

        # 父文档回填（仅 hybrid_traditional 路；不改排名，仅换上下文内容）
        if getattr(self.config, "enable_parent_doc_retrieval", False):
            final_docs = self._attach_parent_documents(final_docs)

        logger.info(
            f"RRF 融合完成：dual={len(dual_docs)} vector={len(vector_docs)} "
            f"bm25={len(bm25_docs)} → 最终 {len(final_docs)} 个文档"
        )
        return final_docs

    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
