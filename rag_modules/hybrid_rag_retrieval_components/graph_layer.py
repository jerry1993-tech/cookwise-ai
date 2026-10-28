"""Neo4j 图读取：图索引构建与节点邻居查询。"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class GraphIndexBuilder:
    """把 data_module 里的实体/关系写进图键值索引。"""

    def __init__(self, driver, graph_indexing, data_module):
        self.driver = driver
        self.graph_indexing = graph_indexing
        self.data_module = data_module
        self.indexed = False

    def build(self):
        """构建图索引"""
        if self.indexed:
            return

        logger.info("开始构建图索引...")

        try:
            # 获取图数据
            recipes = self.data_module.recipes
            ingredients = self.data_module.ingredients
            cooking_steps = self.data_module.cooking_steps

            # 创建实体键值对
            self.graph_indexing.create_entity_key_values(recipes, ingredients, cooking_steps)

            # 创建关系键值对（这里需要从Neo4j获取关系数据）
            relationships = self._extract_relationships()
            self.graph_indexing.create_relation_key_values(relationships)

            # 去重优化
            self.graph_indexing.deduplicate_entities_and_relations()

            self.indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def _extract_relationships(self) -> List[Tuple[str, str, str]]:
        """从Neo4j图中提取关系"""
        relationships = []

        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId >= '200000000' OR target.nodeId >= '200000000'
                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                LIMIT 1000
                """
                result = session.run(query)

                for record in result:
                    relationships.append((
                        record["source_id"],
                        record["relation_type"],
                        record["target_id"]
                    ))

        except Exception as e:
            logger.error(f"提取图关系失败: {e}")

        return relationships


def get_node_neighbors(driver, node_id: str, max_neighbors: int = 3) -> List[str]:
    """获取节点的邻居信息"""
    try:
        with driver.session() as session:
            query = """
            MATCH (n {nodeId: $node_id})-[r]-(neighbor)
            RETURN neighbor.name as name
            LIMIT $limit
            """
            result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
            return [record["name"] for record in result if record["name"]]
    except Exception as e:
        logger.error(f"获取邻居节点失败: {e}")
        return []
