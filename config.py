"""
基于图数据库的RAG系统配置文件
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "all-in-rag")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Milvus配置
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-flash")

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 4096

    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5

    # 父文档回填：RRF 融合后把前几条 chunk 换成整篇菜谱，给生成层更完整的上下文
    enable_parent_doc_retrieval: bool = True
    parent_doc_top_n: int = 3
    parent_doc_max_chars: int = 4000  # 单篇父文档截断长度，避免上下文过长

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度

    def __post_init__(self):
        """初始化后处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,

            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,

            'embedding_model': self.embedding_model,

            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'enable_parent_doc_retrieval': self.enable_parent_doc_retrieval,
            'parent_doc_top_n': self.parent_doc_top_n,
            'parent_doc_max_chars': self.parent_doc_max_chars,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,

            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth
        }


# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig()