"""向量检索：Milvus 召回 + 图邻居信息增强。"""

import logging
from typing import List

from langchain_core.documents import Document

from .graph_layer import get_node_neighbors

logger = logging.getLogger(__name__)


class VectorSearcher:
    """Milvus 向量检索，结果补一跳邻居后转成 Document。"""

    def __init__(self, driver, milvus_module):
        self.driver = driver
        self.milvus_module = milvus_module

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        增强的向量检索：结合图信息
        """
        try:
            # 使用Milvus进行向量检索
            vector_docs = self.milvus_module.similarity_search(query, k=top_k*2)

            # 用图信息增强结果并转换为Document对象
            enhanced_docs = []
            for result in vector_docs:
                # 从Milvus结果创建Document对象
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")

                if node_id:
                    # 从图中获取邻居信息
                    neighbors = get_node_neighbors(self.driver, node_id)
                    if neighbors:
                        # 将邻居信息添加到内容中
                        neighbor_info = f"\n相关信息: {', '.join(neighbors[:3])}"
                        content += neighbor_info

                # 确保recipe_name字段正确设置
                recipe_name = metadata.get("recipe_name", "未知菜品")

                # 调试：打印向量得分
                vector_score = result.get("score", 0.0)
                logger.debug(f"向量检索得分: {recipe_name} = {vector_score}")

                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "recipe_name": recipe_name,  # 确保有recipe_name字段
                        "score": vector_score,
                        "search_type": "vector_enhanced"
                    }
                )
                enhanced_docs.append(doc)

            return enhanced_docs[:top_k]

        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []
