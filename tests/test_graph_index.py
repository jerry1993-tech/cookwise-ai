import sys
from pathlib import Path
from types import SimpleNamespace

# 将项目根目录加入模块搜索路径，支持从任意位置直接运行本脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_modules import GraphIndexingModule


class FakeNode:
    """造一个带 node_id/name/properties 的假节点"""
    def __init__(self, node_id, name=None, properties=None):
        self.node_id = node_id
        self.name = name
        self.properties = properties or {}


def main():
    # 1. 构造假数据
    recipes = [
        FakeNode("r1", "宫保鸡丁", {"category": "川菜", "cuisineType": "川菜", "difficulty": "中等", "cookingTime": "30分钟"}),
        FakeNode("r2", "番茄炒蛋", {"category": "家常菜", "cuisineType": "家常", "difficulty": "简单", "cookingTime": "15分钟"}),
    ]
    ingredients = [
        FakeNode("i1", "鸡胸肉", {"category": "肉类", "nutrition": "高蛋白"}),
        FakeNode("i2", "花生", {"category": "坚果", "nutrition": "高脂肪"}),
        FakeNode("i3", "鸡蛋", {"category": "蛋类", "nutrition": "高蛋白"}),
    ]
    cooking_steps = [
        FakeNode("s1", None, {"description": "鸡肉切丁腌制", "order": 1, "technique": "腌制"}),
        FakeNode("s2", None, {"description": "热油爆炒", "order": 2, "technique": "爆炒"}),
    ]
    relationships = [
        ("r1", "REQUIRES", "i1"),
        ("r1", "REQUIRES", "i2"),
        ("r2", "REQUIRES", "i3"),
        ("r1", "HAS_STEP", "s1"),
        ("r1", "HAS_STEP", "s2"),
    ]

    # 2. 初始化模块（LLM 增强开关关着，所以 llm_client 传 None 也不会被用到）
    config = SimpleNamespace(enable_llm_relation_keys=False)
    module = GraphIndexingModule(config, llm_client=None)

    # 3. 建实体键值对 + 关系键值对
    module.create_entity_key_values(recipes, ingredients, cooking_steps)
    module.create_relation_key_values(relationships)

    print("== 去重前统计 ==")
    print(module.get_statistics())

    # 4. 去重并重建映射
    module.deduplicate_entities_and_relations()
    print("== 去重后统计 ==")
    print(module.get_statistics())

    # 5. 按索引键查询
    entities = module.get_entities_by_key("宫保鸡丁")
    print("按'宫保鸡丁'查到实体:", [e.entity_name for e in entities])

    rels = module.get_relations_by_key("食材搭配")
    print("按'食材搭配'查到关系:", [r.relation_id for r in rels])


if __name__ == "__main__":
    main()