"""双层检索：关键词提取 → 实体级 + 主题级检索 → 合并成 Document。"""

import json
import logging
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document

from .graph_layer import get_node_neighbors

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'entity' or 'topic'
    metadata: Dict[str, Any]


class DualLevelRetriever:
    """
    双层检索范式：
    1. LLM 提取实体级 / 主题级关键词（失败走图谱词典 + jieba 兜底）
    2. 实体级走图键值索引，不足时用 Neo4j 全文索引补
    3. 主题级走关系键值索引 + 分类匹配，不足时用 Neo4j 补
    """

    def __init__(self, driver, graph_indexing, llm_client, config, data_module):
        self.driver = driver
        self.graph_indexing = graph_indexing
        self.llm_client = llm_client
        self.config = config
        self.data_module = data_module

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        提取查询关键词：实体级 + 主题级
        """
        prompt = f"""
        作为烹饪知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
           - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
           - 对于抽象查询，推测相关的具体食材/菜品

        2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
           - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
           - 排除动作词：推荐、介绍、制作、怎么做等

        示例：
        查询："推荐几个减肥菜"
        {{
            "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
            "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
        }}

        查询："川菜有什么特色"
        {{
            "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
            "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                extra_body={"thinking": {"type": "disabled"}},
            )

            result = json.loads(response.choices[0].message.content.strip())
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])

            logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            return entity_keywords, topic_keywords

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 降级方案：图谱实体词典 + jieba 分词；中文不能按空格切
            try:
                from ..query_routing.domain import DOMAIN
                from ..query_routing.text_utils import fallback_keywords

                recipes = [
                    getattr(node, "name", "")
                    for node in (getattr(self.data_module, "recipes", None) or [])
                ]
                ingredients = [
                    getattr(node, "name", "")
                    for node in (getattr(self.data_module, "ingredients", None) or [])
                ]
                topics = [
                    term for terms in DOMAIN.topic_terms.values() for term in terms
                ]
                return fallback_keywords(
                    query,
                    recipes=recipes,
                    ingredients=ingredients,
                    cuisines=DOMAIN.cuisine_terms,
                    aliases=DOMAIN.aliases,
                    topic_terms=topics,
                    generic_names=DOMAIN.generic_entity_names,
                )
            except Exception as fallback_error:
                logger.error(f"关键词兜底也失败，按 jieba 分词: {fallback_error}")
                from ..query_routing.text_utils import segment_words

                keywords = segment_words(str(query))
                return keywords[:3], keywords[3:6]

    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        实体级检索：专注于具体实体和关系
        使用图索引的键值对结构进行检索
        """
        results = []

        # 1. 使用图索引进行实体检索
        for keyword in entity_keywords:
            # 检索匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)

            for entity in entities:
                # 获取邻居信息
                neighbors = get_node_neighbors(self.driver, entity.metadata["node_id"], max_neighbors=2)

                # 构建增强内容
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"

                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword
                    }
                ))

        # 2. 如果图索引结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)

        # 3. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j补充检索"""
        results = []

        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                CALL db.index.fulltext.queryNodes('recipe_fulltext_index', keyword + '*')
                YIELD node, score
                WHERE node:Recipe
                RETURN
                    node.nodeId as node_id,
                    node.name as name,
                    node.description as description,
                    labels(node) as labels,
                    score
                ORDER BY score DESC
                LIMIT $limit
                """

                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })

                for record in result:
                    content_parts = []
                    if record["name"]:
                        content_parts.append(f"菜品: {record['name']}")
                    if record["description"]:
                        content_parts.append(f"描述: {record['description']}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=float(record["score"]) * 0.7,  # 补充检索得分较低
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"
                        }
                    ))

        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")

        return results

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        主题级检索：专注于广泛主题和概念
        使用图索引的关系键值对结构进行主题检索
        """
        results = []

        # 1. 使用图索引进行关系/主题检索
        for keyword in topic_keywords:
            # 检索匹配的关系
            relations = self.graph_indexing.get_relations_by_key(keyword)

            for relation in relations:
                # 获取相关实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)

                if source_entity and target_entity:
                    # 构建丰富的主题内容
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关菜品: {source_entity.entity_name}",
                        f"相关信息: {target_entity.entity_name}"
                    ]

                    # 添加源实体的详细信息
                    if source_entity.entity_type == "Recipe":
                        newline = '\n'
                        content_parts.append(f"菜品详情: {source_entity.value_content.split(newline)[0]}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=relation.source_entity,  # 以主要实体为ID
                        node_type=source_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))

        # 2. 使用实体的分类信息进行主题检索
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                if entity.entity_type == "Recipe":
                    # 构建分类主题内容
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content
                    ]

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match"
                        }
                    ))

        # 3. 如果结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)

        # 4. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j主题级检索补充"""
        results = []

        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (r:Recipe)
                WHERE r.category CONTAINS keyword
                   OR r.cuisineType CONTAINS keyword
                   OR r.tags CONTAINS keyword
                WITH r, keyword
                OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
                WITH r, keyword, collect(i.name)[0..3] as ingredients
                RETURN
                    r.nodeId as node_id,
                    r.name as name,
                    r.category as category,
                    r.cuisineType as cuisine_type,
                    r.difficulty as difficulty,
                    ingredients,
                    keyword as matched_keyword
                ORDER BY r.difficulty ASC, r.name
                LIMIT $limit
                """

                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })

                for record in result:
                    content_parts = []
                    content_parts.append(f"菜品: {record['name']}")

                    if record["category"]:
                        content_parts.append(f"分类: {record['category']}")
                    if record["cuisine_type"]:
                        content_parts.append(f"菜系: {record['cuisine_type']}")
                    if record["difficulty"]:
                        content_parts.append(f"难度: {record['difficulty']}")

                    if record["ingredients"]:
                        ingredients_str = ', '.join(record["ingredients"][:3])
                        content_parts.append(f"主要食材: {ingredients_str}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=0.75,  # 补充检索得分
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "category": record["category"],
                            "cuisine_type": record["cuisine_type"],
                            "difficulty": record["difficulty"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))

        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")

        return results

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """
        双层检索：结合实体级和主题级检索
        """
        logger.info(f"开始双层检索: {query}")

        # 1. 提取关键词
        entity_keywords, topic_keywords = self.extract_query_keywords(query)

        # 2. 执行双层检索
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)

        # 3. 结果合并和排序
        all_results = entity_results + topic_results

        # 4. 去重和重排序
        seen_nodes = set()
        unique_results = []

        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        # 5. 转换为Document格式
        documents = []
        for result in unique_results[:top_k]:
            # 确保recipe_name字段正确设置
            recipe_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知菜品")

            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "recipe_name": recipe_name,  # 确保有recipe_name字段
                    "search_type": "dual_level",  # 设置搜索类型
                    **result.metadata
                }
            )
            documents.append(doc)

        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents
