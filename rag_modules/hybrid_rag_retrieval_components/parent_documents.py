"""父文档回填：把命中的 chunk 换回它所属的整篇菜谱。"""

import logging
from typing import Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def build_parent_doc_map(data_module) -> Dict[str, Document]:
    """{str(node_id): 整篇父菜谱 Document}，由分块前的 data_module.documents 懒建一次。"""
    docs = getattr(data_module, "documents", None) or []
    m: Dict[str, Document] = {}
    for d in docs:
        nid = d.metadata.get("node_id")
        if nid is not None:
            m[str(nid)] = d
    return m


def attach_parent_documents(
    docs: List[Document],
    parent_doc_map: Dict[str, Document],
    top_n: int,
    max_chars: int,
) -> List[Document]:
    """RRF 去重后，前 top_n 条且能在映射中找到父菜谱的，
    用整篇父菜谱（超 max_chars 截断）替换 chunk；其余原样不变。
    不改顺序/数量/排名，不 mutate 输入（被替换的造新 Document，未替换的直接传原对象）。"""
    if not parent_doc_map:
        logger.warning("父文档映射为空（data_module.documents 未就绪），父文档回填未生效，仍然使用原chunk填充上下文")
        return docs

    out: List[Document] = []
    for i, doc in enumerate(docs):
        if i >= top_n:
            out.append(doc)
            continue
        nid = doc.metadata.get("node_id")
        key = str(nid if nid is not None else doc.metadata.get("parent_id"))
        parent = parent_doc_map.get(key)
        if parent is None:
            out.append(doc)
            continue
        pc = parent.page_content or ""
        if len(pc) > max_chars:
            pc = pc[:max_chars] + "…（父文档已截断）"
        out.append(Document(page_content=pc, metadata=dict(doc.metadata)))
    return out
