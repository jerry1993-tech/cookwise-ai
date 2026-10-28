"""RRF（Reciprocal Rank Fusion）融合：把多路排名合成一个排名。"""

import hashlib
from typing import Dict, List, Tuple

from langchain_core.documents import Document

# RRF 融合的常数 k：Cormack et al. 2009 默认值
RRF_K = 60


def rrf_merge(
    ranked_lists: List[Tuple[str, List[Document]]],
    top_k: int,
    k: int = RRF_K,
) -> List[Document]:
    """
    Reciprocal Rank Fusion: score(d) = Σ_i 1 / (k + best_rank_i(d))

    Args:
        ranked_lists: 多路 (source_name, ranked_docs) — docs 按相关度降序
        top_k: 最终返回个数
        k: RRF 平滑常数，默认 60（Cormack et al. 2009）

    去重 key：node_id 优先，page_content[:200] hash 兜底。

    同 source 内同 doc_id 多次命中（如一道菜的多个 chunk 共享 recipe.nodeId）：
        - 算分只取该 source 内最佳 rank（最小 rank）一次，避免重复加分
        - 命中 chunk 数另存到 rrf_chunk_hits，供后续分析

    canonical doc（最终展示给 LLM 的 page_content）：
        选全局最小 rank 那个 chunk；rank 相同时按 ranked_lists 顺序优先。

    返回的 Document 是新对象，不会 mutate 输入 list 里的 Document。
    """
    # doc_id -> source_name -> 该 source 内最小 rank（用于算分）
    best_rank_per_source: Dict[str, Dict[str, int]] = {}
    # doc_id -> source_name -> 该 source 内命中 chunk 次数（信息存档）
    chunk_hits_per_source: Dict[str, Dict[str, int]] = {}
    # doc_id -> (global_best_rank, source_priority, doc) — 选 canonical doc
    best_doc_info: Dict[str, Tuple[int, int, Document]] = {}

    for source_priority, (source_name, ranked_docs) in enumerate(ranked_lists):
        for rank, doc in enumerate(ranked_docs, start=1):
            node_id = doc.metadata.get("node_id")
            doc_id = (
                str(node_id) if node_id is not None
                else f"hash::{hashlib.md5(doc.page_content[:200].encode('utf-8')).hexdigest()}"
            )

            if doc_id not in best_rank_per_source:
                best_rank_per_source[doc_id] = {}
                chunk_hits_per_source[doc_id] = {}

            curr_best = best_rank_per_source[doc_id].get(source_name)
            # 如果是第一次出现或者当前rank比记录的更小，则更新
            if curr_best is None or rank < curr_best:
                best_rank_per_source[doc_id][source_name] = rank

            chunk_hits_per_source[doc_id][source_name] = (
                chunk_hits_per_source[doc_id].get(source_name, 0) + 1
            )

            new_key = (rank, source_priority)
            if (
                doc_id not in best_doc_info
                or new_key < (best_doc_info[doc_id][0], best_doc_info[doc_id][1])
            ):
                best_doc_info[doc_id] = (rank, source_priority, doc)

    # 每个 source 只用 best rank 算一次贡献
    rrf_scores: Dict[str, float] = {
        doc_id: sum(1.0 / (k + r) for r in source_ranks.values())
        for doc_id, source_ranks in best_rank_per_source.items()
    }

    sorted_ids = sorted(
        rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True
    )

    merged: List[Document] = []
    for doc_id in sorted_ids[:top_k]:
        _, _, source_doc = best_doc_info[doc_id]
        # 浅 copy metadata，避免 mutate 上游 Document
        new_metadata = dict(source_doc.metadata)
        new_metadata["rrf_score"] = rrf_scores[doc_id]
        new_metadata["rrf_sources"] = list(best_rank_per_source[doc_id].keys())
        new_metadata["rrf_ranks"] = dict(best_rank_per_source[doc_id])
        new_metadata["rrf_chunk_hits"] = dict(chunk_hits_per_source[doc_id])
        new_metadata["final_score"] = rrf_scores[doc_id]
        merged.append(Document(
            page_content=source_doc.page_content,
            metadata=new_metadata,
        ))

    return merged
