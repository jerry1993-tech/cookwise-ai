"""BM25 索引：建索引与按 query 检索。"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .tokenizer import tokenize_chinese

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 索引 + 原始文档（按索引位置对齐）。"""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_docs: List[Document] = []

    def build(self, chunks: List[Document]):
        """用全量 chunk 建索引（jieba 分词 + 中文停用词过滤）"""
        if not chunks:
            return
        self.corpus_docs = list(chunks)
        tokenized_corpus = [tokenize_chinese(d.page_content) for d in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        avg_tokens = sum(len(t) for t in tokenized_corpus) / max(1, len(tokenized_corpus))
        logger.info(
            f"BM25(jieba+stopwords) 索引构建完成，文档数: {len(chunks)}，"
            f"平均 token 数: {avg_tokens:.1f}"
        )

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        BM25 检索：jieba 分词后查 BM25Okapi 索引，按分数降序返回 top_k。
        分数写入 metadata["bm25_score"]，供调试与未来潜在的分数级融合使用。
        """
        if self.bm25 is None or not self.corpus_docs:
            logger.warning("BM25 索引未初始化，bm25_search 返回空")
            return []

        tokenized_query = tokenize_chinese(query)
        if not tokenized_query:
            logger.debug(f"BM25 query 分词为空，跳过: {query}")
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # 按分数降序取 top_k 索引
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        docs: List[Document] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                # BM25 分数 ≤ 0 视为无关（IDF/TF 全无贡献），不进结果
                continue
            src = self.corpus_docs[idx]
            recipe_name = (
                src.metadata.get("recipe_name")
                or src.metadata.get("name")
                or "未知菜品"
            )
            doc = Document(
                page_content=src.page_content,
                metadata={
                    **src.metadata,
                    "recipe_name": recipe_name,
                    "search_method": "bm25",
                    "search_type": "bm25",
                    "bm25_score": score,
                }
            )
            docs.append(doc)

        logger.info(f"BM25 检索完成，返回 {len(docs)} 个文档（query tokens={tokenized_query}）")
        return docs
