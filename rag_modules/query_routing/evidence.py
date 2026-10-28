"""
执行层：按 RoutePlan 去正确的证据源取证据。

- GraphEvidenceAdapter：实体绑定后的 Neo4j 精确查询 + 菜谱原文读取
- TextEvidenceAdapter：现有 hybrid 检索
- EvidenceService：按 engine 分发、合并、加元数据

图谱的公开签名不在本文件里改；这里只读 Neo4j。
"""

import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from .domain import CUISINE_TERMS, DOMAIN, GENERIC_ENTITY_NAMES
from .schemas import SubPlan
from .text_utils import normalize_text
from .tips_knowledge import MealPlan, TipsKnowledgeBase

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 候选过滤字段 -> Cypher 片段；$param 会在 _run_candidate_query 里换成参数名。
CANDIDATE_FIELD_TEMPLATES: Dict[str, str] = {
    "name": "coalesce(r.name,'') CONTAINS $param",
    "tags": "coalesce(r.tags,'') CONTAINS $param",
    "category": "coalesce(r.category,'') CONTAINS $param",
    "cuisine": "toLower(coalesce(r.cuisineType,'')) CONTAINS toLower($param)",
    "ingredient": (
        "any(n IN [(r)-[:REQUIRES]->(i:Ingredient) | i.name] "
        "WHERE n CONTAINS $param)"
    ),
}


def _document(content: str, metadata: Dict[str, Any]) -> Document:
    """统一构造 Document，顺便补上旧代码会读的字段。"""
    metadata.setdefault("search_type", metadata.get("source", "structured"))
    metadata.setdefault("final_score", 1.0)
    return Document(page_content=content.strip(), metadata=metadata)


def _collapse(text: str) -> str:
    """把多行空白压成一行，方便放进证据块。"""
    return re.sub(r"\s+", " ", text or "").strip()


class GraphEvidenceAdapter:
    """CookWise Neo4j + 菜谱 Markdown 的只读适配器。"""

    def __init__(self, graph_retriever=None, data_module=None, project_root: Optional[Path] = None):
        self.graph_retriever = graph_retriever
        self.data_module = data_module or getattr(graph_retriever, "data_module", None)
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.data_root = self.project_root / "data"
        self.domain = DOMAIN

        self._recipe_names: Optional[List[str]] = None
        self._ingredient_names: Optional[List[str]] = None
        self._cuisines: Optional[List[str]] = None
        self._markdown_cache: Dict[str, str] = {}
        self._file_index: Optional[Dict[str, Path]] = None

    @property
    def driver(self):
        """动态读取 driver：GraphRAGRetrieval.initialize() 可能在路由器之后才执行。"""
        return getattr(self.graph_retriever, "driver", None)

    @property
    def available(self) -> bool:
        """是否有可用的 Neo4j 连接。"""
        return self.driver is not None

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------

    def _run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行只读 Cypher；任何失败都返回空列表。"""
        if not self.available:
            return []
        try:
            with self.driver.session() as session:
                return [dict(record) for record in session.run(cypher, params or {})]
        except Exception as exc:
            logger.warning("Neo4j 查询失败: %s", exc)
            return []

    def get_recipe_names(self) -> List[str]:
        """全部菜名；用于实体词典和规则兜底。"""
        if self._recipe_names is not None:
            return self._recipe_names

        rows = self._run("MATCH (r:Recipe) WHERE r.name IS NOT NULL RETURN r.name AS name ORDER BY r.name")
        names = [str(row["name"]) for row in rows if row.get("name")]
        if not names:
            names = [
                str(node.name)
                for node in (getattr(self.data_module, "recipes", None) or [])
                if getattr(node, "name", "")
            ]
        names = [name for name in names if name not in GENERIC_ENTITY_NAMES]
        if names:
            self._recipe_names = names
        return names

    def get_ingredient_names(self) -> List[str]:
        """全部食材名；只用于实体链接，不用于回答清单。"""
        if self._ingredient_names is not None:
            return self._ingredient_names

        rows = self._run(
            "MATCH (i:Ingredient) WHERE i.name IS NOT NULL "
            "RETURN DISTINCT i.name AS name ORDER BY i.name LIMIT 3000"
        )
        names = [str(row["name"]) for row in rows if row.get("name")]
        if not names:
            names = [
                str(node.name)
                for node in (getattr(self.data_module, "ingredients", None) or [])
                if getattr(node, "name", "")
            ]
        names = [name for name in names if name not in GENERIC_ENTITY_NAMES]
        if names:
            self._ingredient_names = names
        return names

    def get_cuisines(self) -> List[str]:
        """图谱里真实出现的菜系。"""
        if self._cuisines is not None:
            return self._cuisines

        rows = self._run(
            "MATCH (r:Recipe) WHERE r.cuisineType IS NOT NULL "
            "RETURN DISTINCT r.cuisineType AS cuisine"
        )
        cuisines: List[str] = []
        for row in rows:
            for part in re.split(r"[,，、/|;；]+", str(row.get("cuisine") or "")):
                part = part.strip()
                if part and part not in cuisines:
                    cuisines.append(part)
        if cuisines:
            self._cuisines = cuisines
            return self._cuisines
        return list(CUISINE_TERMS)

    def resolve_recipe(self, name: str) -> Tuple[str, Dict[str, Any], List[str]]:
        """把用户说的菜名落到图谱里的真实菜名。"""
        name = normalize_text(name)
        if not name:
            return "", {}, []

        if not self.available:
            for node in (getattr(self.data_module, "recipes", None) or []):
                if getattr(node, "name", "") == name:
                    return name, dict(getattr(node, "properties", {}) or {}), []
            return name, {}, []

        rows = self._run(
            """
            MATCH (r:Recipe)
            WHERE r.name = $name OR r.name CONTAINS $name OR $name CONTAINS r.name
            RETURN r.name AS name, properties(r) AS props
            ORDER BY CASE WHEN r.name = $name THEN 0 ELSE 1 END, size(r.name)
            LIMIT 5
            """,
            {"name": name},
        )
        if not rows:
            return name, {}, []

        best = rows[0]
        alternatives = [
            str(row["name"]) for row in rows[1:]
            if row.get("name") and row.get("name") != best.get("name")
        ]
        return str(best.get("name") or name), dict(best.get("props") or {}), alternatives

    # ------------------------------------------------------------------
    # 菜谱原文
    # ------------------------------------------------------------------

    def get_markdown(self, recipe_name: str) -> str:
        """读取菜谱 Markdown；优先 filePath，其次 data_module 文档，最后按文件名找。"""
        name, props, _ = self.resolve_recipe(recipe_name)
        if not name:
            return ""
        if name in self._markdown_cache:
            return self._markdown_cache[name]

        text = ""
        file_path = props.get("filePath") or props.get("filepath")
        if file_path:
            rel = str(file_path).replace("\\", "/").lstrip("/")
            if rel.startswith("data/"):
                rel = rel[5:]
            path = self.data_root / rel
            text = self._read(path)

        if not text:
            for doc in (getattr(self.data_module, "documents", None) or []):
                if doc.metadata.get("recipe_name") == name:
                    text = doc.page_content
                    break

        if not text:
            if self._file_index is None:
                self._file_index = {
                    path.stem: path for path in self.data_root.rglob("*.md")
                }
            path = self._file_index.get(name)
            if path:
                text = self._read(path)

        self._markdown_cache[name] = text
        return text

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    @staticmethod
    def get_section(markdown: str, title_terms: Sequence[str]) -> str:
        """按二级标题取一节，例如“操作”“计算”“必备原料和工具”。"""
        if not markdown:
            return ""
        matches = list(re.finditer(r"^##\s+(.+)$", markdown, flags=re.MULTILINE))
        for index, match in enumerate(matches):
            title = match.group(1).strip()
            if any(term in title for term in title_terms):
                start = match.start()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
                return markdown[start:end].strip()
        return ""

    @staticmethod
    def find_measure(markdown: str, query: str) -> Optional[Tuple[str, str, str]]:
        """按问题类型从原文里找精确数值：热量、时间或用量。"""
        if not markdown:
            return None

        patterns: List[Tuple[str, str]] = []
        if any(term in query for term in ("千卡", "大卡", "热量", "卡路里", "能量")):
            patterns.append((r"(\d+(?:\.\d+)?)\s*(千卡|大卡|kcal|卡路里)", "kcal"))
        if any(term in query for term in ("多少克", "几克", "克数", "毫升", "多少毫升")):
            patterns.append((r"(\d+(?:\.\d+)?)\s*(克|g|毫升|ml)", "amount"))
        if any(term in query for term in ("分钟", "多久", "时间", "火候", "蒸", "煮")):
            patterns.append((r"(\d+(?:\.\d+)?)\s*(分钟|小时)", "time"))

        # 明确问热量/用量时，不能拿菜名里的“蒸/煮”触发的时间数字顶替
        if any(unit_type == "kcal" for _, unit_type in patterns):
            patterns = [item for item in patterns if item[1] == "kcal"]
        elif any(unit_type == "amount" for _, unit_type in patterns):
            patterns = [item for item in patterns if item[1] == "amount"]

        # 问蛋白质/脂肪等图谱没有的指标时不要退回温度/时间数字，交给 no_data
        for pattern, unit_type in patterns:
            match = re.search(pattern, markdown, flags=re.IGNORECASE)
            if match:
                value = f"{match.group(1)} {match.group(2)}"
                sentence = _collapse(markdown[max(0, match.start() - 60): match.end() + 60])
                return value, unit_type, sentence
        return None

    # ------------------------------------------------------------------
    # 证据文档
    # ------------------------------------------------------------------

    def ingredients_docs(self, recipe_name: str, query: str) -> List[Document]:
        """图谱 REQUIRES 全量 + 原文的“必备原料/计算”两节。"""
        name, props, alternatives = self.resolve_recipe(recipe_name)
        if not name:
            return []

        rows = self._run(
            """
            MATCH (r:Recipe {name: $name})-[req:REQUIRES]->(i:Ingredient)
            RETURN i.name AS name, req.amount AS amount, req.unit AS unit,
                   i.category AS category
            ORDER BY i.name
            """,
            {"name": name},
        )

        docs: List[Document] = []
        if rows:
            lines = [f"【菜谱】{name}", f"【完整食材清单（图谱 REQUIRES，共 {len(rows)} 项）】"]
            for index, row in enumerate(rows, 1):
                amount = str(row.get("amount") or "").strip()
                unit = str(row.get("unit") or "").strip()
                suffix = f" {amount}{unit}" if amount or unit else ""
                lines.append(f"{index}. {row.get('name', '')}{suffix}")

            lines.append("【说明】以上是图谱中的全部食材；原文标注的“可选/进阶”项可能也在其中。")
            docs.append(_document("\n".join(lines), {
                "source": "graph:REQUIRES",
                "evidence_type": "ingredients",
                "recipe_name": name,
                "expected_count": len(rows),
                "found_count": len(rows),
                "alternatives": alternatives,
            }))

        markdown = self.get_markdown(name)
        required = self.get_section(markdown, ("必备原料", "原料和工具"))
        calculation = self.get_section(markdown, ("计算",))
        if required or calculation:
            docs.append(_document(
                "\n\n".join(part for part in (required, calculation) if part),
                {
                    "source": "recipe_text:ingredients",
                    "evidence_type": "recipe_text",
                    "recipe_name": name,
                },
            ))

        return docs

    def steps_docs(self, recipe_name: str, query: str) -> List[Document]:
        """图谱步骤 + 原文“操作” + 火候/时间元数据。"""
        name, props, alternatives = self.resolve_recipe(recipe_name)
        if not name:
            return []

        rows = self._run(
            """
            MATCH (r:Recipe {name: $name})-[c:CONTAINS_STEP]->(s:CookingStep)
            RETURN s.name AS name, s.description AS description,
                   s.stepNumber AS step_number, s.timeEstimate AS time_estimate,
                   c.stepOrder AS step_order
            ORDER BY coalesce(c.stepOrder, s.stepNumber, 999)
            """,
            {"name": name},
        )

        docs: List[Document] = []
        if rows:
            lines = [f"【菜谱】{name}", f"【图谱步骤（共 {len(rows)} 步）】"]
            for index, row in enumerate(rows, 1):
                order = row.get("step_order") or row.get("step_number") or index
                if isinstance(order, float) and order.is_integer():
                    order = int(order)
                title = str(row.get("name") or "").strip()
                lines.append(f"{order}. {title}")
                if row.get("description"):
                    lines.append(f"   描述：{_collapse(str(row['description']))}")
                if row.get("time_estimate"):
                    lines.append(f"   时间：{row['time_estimate']}")
            docs.append(_document("\n".join(lines), {
                "source": "graph:CONTAINS_STEP",
                "evidence_type": "steps",
                "recipe_name": name,
                "expected_count": len(rows),
                "found_count": len(rows),
            }))

        markdown = self.get_markdown(name)
        operation = self.get_section(markdown, ("操作", "制作步骤"))
        if operation:
            docs.append(_document(operation, {
                "source": "recipe_text:steps",
                "evidence_type": "recipe_text",
                "recipe_name": name,
            }))

        time_info = []
        labels = {
            "prepTime": "准备时间", "cookTime": "烹饪时间", "servings": "份量",
            "difficulty": "难度", "cuisineType": "菜系", "tags": "标签",
        }
        for key, label in labels.items():
            if props.get(key):
                time_info.append(f"{label}：{props[key]}")
        if time_info:
            docs.append(_document(
                f"【菜谱】{name}\n" + "\n".join(time_info),
                {
                    "source": "graph:Recipe",
                    "evidence_type": "time_meta",
                    "recipe_name": name,
                },
            ))

        return docs

    def overview_docs(self, recipe_name: str, query: str) -> List[Document]:
        """菜谱属性 + 原文开头。"""
        name, props, alternatives = self.resolve_recipe(recipe_name)
        if not name:
            return []

        lines = [f"【菜谱】{name}"]
        labels = {
            "category": "分类",
            "cuisineType": "菜系",
            "difficulty": "难度",
            "prepTime": "准备时间",
            "cookTime": "烹饪时间",
            "servings": "份量",
            "tags": "标签",
        }
        for key, label in labels.items():
            if props.get(key):
                lines.append(f"{label}：{props[key]}")

        docs = [_document("\n".join(lines), {
            "source": "graph:Recipe",
            "evidence_type": "overview",
            "recipe_name": name,
            "alternatives": alternatives,
        })]

        markdown = self.get_markdown(name)
        intro = markdown.split("\n## ", 1)[0].strip() if markdown else ""
        if intro:
            docs.append(_document(intro, {
                "source": "recipe_text:intro",
                "evidence_type": "recipe_text",
                "recipe_name": name,
            }))
        return docs

    def value_docs(self, recipe_name: str, query: str) -> List[Document]:
        """精确数值：只返回原文里真实出现的那一句。"""
        name, props, _ = self.resolve_recipe(recipe_name)
        if not name:
            return []

        markdown = self.get_markdown(name)
        measure = self.find_measure(markdown, query)
        if not measure:
            return []

        value, unit_type, sentence = measure
        content = (
            f"【菜谱】{name}\n"
            f"【原文数值】{value}\n"
            f"【原文上下文】{sentence}\n"
            f"【说明】以上数值来自菜谱原文，未做换算或估算。"
        )
        return [_document(content, {
            "source": "recipe_text:value",
            "evidence_type": "value",
            "recipe_name": name,
            "value_found": True,
            "value": value,
            "value_type": unit_type,
            "value_policy": "exact_only",
        })]

    def benefit_docs(self, recipe_name: str, query: str) -> List[Document]:
        """营养/功效：返回原文开头，不做医学推断。"""
        name, props, _ = self.resolve_recipe(recipe_name)
        if not name:
            return []

        markdown = self.get_markdown(name)
        intro = markdown.split("\n## ", 1)[0].strip() if markdown else ""
        if not intro:
            return []
        return [_document(intro[:1200], {
            "source": "recipe_text:benefit",
            "evidence_type": "benefit",
            "recipe_name": name,
        })]

    # ------------------------------------------------------------------
    # 推荐候选
    # ------------------------------------------------------------------

    def candidate_docs(self, slots: Dict[str, Any], limit: int) -> List[Document]:
        """按槽位过滤菜谱，返回候选列表。"""
        rows, note = self._query_candidates(slots, limit)
        if not rows:
            return []

        lines = [f"【候选菜谱（共 {len(rows)} 道）】"]
        for row in rows:
            lines.append(self._candidate_line(row))
        if note:
            lines.append(f"【说明】{note}")

        return [_document("\n".join(lines), {
            "source": "graph:Recipe:candidates",
            "evidence_type": "candidates",
            "candidate_count": len(rows),
            "slots": slots,
        })]

    def meal_candidates_docs(self, plan: MealPlan, limit: int) -> List[Document]:
        """按配菜规则取荤菜/素菜/鱼样本。"""
        docs: List[Document] = []
        groups = [
            ("荤菜", plan.meat + 2),
            ("素菜", plan.veg + 2),
        ]
        if plan.add_fish:
            groups.append(("水产", 2))

        for category, group_limit in groups:
            rows, _ = self._query_candidates({"category": category}, max(2, group_limit))
            if not rows:
                continue
            lines = [f"【{category}候选（取前 {len(rows)} 道）】"]
            lines.extend(self._candidate_line(row) for row in rows)
            docs.append(_document("\n".join(lines), {
                "source": "graph:Recipe:meal_candidates",
                "evidence_type": "meal_candidates",
                "category": category,
                "candidate_count": len(rows),
            }))
        return docs

    def _query_candidates(
        self, slots: Dict[str, Any], limit: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        """拼装候选查询；辣度过滤无结果时放宽一次并说明。"""
        rows = self._run_candidate_query(slots, limit)
        spicy = str(slots.get("spicy") or "").strip()
        known = {value for value, _ in self.domain.slot_vocabulary.spicy_aliases}
        if rows or spicy not in known:
            return rows, ""

        relaxed = dict(slots)
        relaxed.pop("spicy", None)
        rows = self._run_candidate_query(relaxed, limit)
        if rows:
            return rows, f"没有找到符合“{spicy}”的候选，以下为放宽辣度后的结果。"
        return [], ""

    def _run_candidate_query(self, slots: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """按 domain.CANDIDATE_FILTERS 生成候选查询，新维度不用改这里。"""
        conditions, params, rank_terms, ingredient_terms = self._collect_candidate_filters(slots)
        params["limit"] = max(1, int(limit))
        cypher = self._build_candidate_cypher(conditions, params, rank_terms, ingredient_terms)
        return self._run(cypher, params)

    def _collect_candidate_filters(
        self, slots: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any], List[str], List[str]]:
        """把槽位翻译成 Cypher 条件、参数和排序词。"""
        conditions = ["r.name IS NOT NULL"]
        params: Dict[str, Any] = {}
        rank_terms: List[str] = []
        ingredient_terms: List[str] = []
        param_index = 0

        for spec in self.domain.candidate_filters:
            raw = slots.get(spec.slot)
            values = raw if spec.multi_value and isinstance(raw, (list, tuple)) else [raw]
            for value in values:
                if not spec.applies(value):
                    continue
                terms = [term for term in spec.terms_for(value) if term]
                if spec.expand_aliases:
                    terms = [
                        alias
                        for term in terms
                        for alias in self.domain.aliases.get(term, (term,))
                    ]
                    terms = list(dict.fromkeys(terms))
                if not terms:
                    continue

                clauses = []
                for field_name in spec.fields:
                    template = CANDIDATE_FIELD_TEMPLATES[field_name]
                    for term in terms:
                        param_index += 1
                        param_name = f"filter_{param_index}"
                        params[param_name] = term
                        clauses.append(template.replace("$param", f"${param_name}"))

                condition = "(" + " OR ".join(clauses) + ")"
                if spec.is_negative(value):
                    conditions.append(f"NOT {condition}")
                    continue
                conditions.append(condition)
                rank_terms.extend(terms)
                if "ingredient" in spec.fields:
                    ingredient_terms.extend(terms)

        return (
            conditions,
            params,
            list(dict.fromkeys(rank_terms)),
            list(dict.fromkeys(ingredient_terms)),
        )

    def _build_candidate_cypher(
        self,
        conditions: List[str],
        params: Dict[str, Any],
        rank_terms: List[str],
        ingredient_terms: List[str],
    ) -> str:
        """拼装候选查询；字段模板见 CANDIDATE_FIELD_TEMPLATES。"""
        matched_fields = "[] AS matched_ingredients"
        if ingredient_terms:
            params["ingredient_terms"] = ingredient_terms
            matched_fields = (
                "[n IN [(r)-[:REQUIRES]->(i:Ingredient) | i.name] "
                "WHERE any(x IN $ingredient_terms WHERE n CONTAINS x)][0..3] "
                "AS matched_ingredients"
            )

        rank_field = "3 AS match_rank"
        if rank_terms:
            params["rank_terms"] = rank_terms
            rank_field = (
                "CASE "
                "WHEN any(t IN $rank_terms WHERE coalesce(r.name,'') CONTAINS t) THEN 0 "
                "WHEN any(t IN $rank_terms WHERE coalesce(r.tags,'') CONTAINS t) THEN 1 "
                "WHEN any(t IN $rank_terms WHERE coalesce(r.category,'') CONTAINS t) THEN 2 "
                "ELSE 3 END AS match_rank"
            )

        return f"""
        MATCH (r:Recipe)
        WHERE {' AND '.join(conditions)}
        RETURN r.name AS name, r.category AS category, r.cuisineType AS cuisine,
               r.difficulty AS difficulty, r.prepTime AS prep, r.cookTime AS cook,
               r.servings AS servings, r.tags AS tags,
               {matched_fields},
               {rank_field}
        ORDER BY match_rank ASC, r.difficulty ASC, r.name
        LIMIT $limit
        """

    @staticmethod
    def _candidate_line(row: Dict[str, Any]) -> str:
        """把一行候选压成一句可读文本。"""
        name = row.get("name") or "未知菜谱"
        meta = []
        if row.get("cuisine"):
            meta.append(str(row["cuisine"]))
        if row.get("category"):
            meta.append(str(row["category"]))
        if row.get("difficulty") is not None:
            difficulty = row["difficulty"]
            if isinstance(difficulty, float) and difficulty.is_integer():
                difficulty = int(difficulty)
            meta.append(f"难度{difficulty}")
        if row.get("prep"):
            meta.append(f"准备{row['prep']}")
        if row.get("cook"):
            meta.append(f"烹饪{row['cook']}")
        tags = str(row.get("tags") or "").strip()
        matched = [str(item) for item in (row.get("matched_ingredients") or []) if item]
        meta_text = "，".join(meta)
        line = f"- {name}" + (f"（{meta_text}）" if meta_text else "")
        line += f"；标签：{tags}" if tags else ""
        line += f"；命中食材：{'、'.join(matched)}" if matched else ""
        return line


class TextEvidenceAdapter:
    """包装现有 hybrid 检索，不改变它的公开签名。"""

    def __init__(self, traditional_retrieval, graph_adapter: Optional[GraphEvidenceAdapter] = None):
        self.traditional_retrieval = traditional_retrieval
        self.graph_adapter = graph_adapter

    def search(self, query: str, top_k: int) -> List[Document]:
        """普通混合检索。"""
        if self.traditional_retrieval is None or not hasattr(self.traditional_retrieval, "hybrid_search"):
            return []
        try:
            docs = self.traditional_retrieval.hybrid_search(query, top_k)
        except Exception as exc:
            logger.warning("hybrid 检索失败: %s", exc)
            return []

        for doc in docs:
            doc.metadata.setdefault("source", "hybrid")
            doc.metadata.setdefault("evidence_type", "text")
        return docs

    def recipe_search(self, recipe_name: str, query: str, top_k: int) -> List[Document]:
        """带菜名的文本检索；尽量只保留该菜谱的块。"""
        if not recipe_name:
            return self.search(query, top_k)

        docs = self.search(f"{recipe_name} {query}", top_k * 2)
        matched = [
            doc for doc in docs
            if doc.metadata.get("recipe_name") == recipe_name or recipe_name in doc.page_content
        ]
        return (matched or docs)[:top_k]


class EvidenceService:
    """按子计划分发到正确证据源。"""

    def __init__(
        self,
        graph_adapter: GraphEvidenceAdapter,
        text_adapter: TextEvidenceAdapter,
        tips: TipsKnowledgeBase,
        graph_retriever=None,
    ):
        self.graph = graph_adapter
        self.text = text_adapter
        self.tips = tips
        self.graph_retriever = graph_retriever

    def execute(self, sub_plan: SubPlan) -> List[Document]:
        """执行一个子计划，返回带来源元数据的文档。"""
        intent = sub_plan.intent
        route_key = intent.route_key
        engine = sub_plan.engine
        query = sub_plan.query
        slots = intent.slots or {}
        recipe_name = self._recipe_name(intent)

        docs: List[Document] = []

        if engine == "none":
            docs = []
        elif engine == "tips_pairing":
            docs = self._pairing_docs(query)
        elif engine == "tips_safety":
            docs = self._safety_docs(query)
        elif engine == "tips_technique":
            docs = self._technique_docs(query)
            docs.extend(self.text.search(query, sub_plan.top_k))
        elif engine == "tips_rule":
            docs = self._scenario_docs(query, slots)
        elif engine == "graph_relation":
            docs = self._graph_relation_docs(query, sub_plan.top_k)
        elif engine == "hybrid":
            docs = self.text.search(query, sub_plan.top_k)
        elif engine == "combined":
            docs = self.combined_search(query, sub_plan.top_k)
        elif engine == "graph_entity":
            docs = self._graph_entity_docs(route_key, recipe_name, query, slots, sub_plan.top_k)
            if not docs:
                docs = self.text.search(query, sub_plan.top_k)
        else:
            docs = self.text.search(query, sub_plan.top_k)

        for doc in docs:
            doc.metadata.setdefault("route_key", route_key)
            doc.metadata.setdefault("engine", engine)
            doc.metadata.setdefault("sub_query_index", sub_plan.index)
            doc.metadata.setdefault("sub_query", query)
            if recipe_name:
                doc.metadata.setdefault("recipe_name", recipe_name)

        return docs

    # ------------------------------------------------------------------
    # engine 分发
    # ------------------------------------------------------------------

    def _graph_entity_docs(
        self,
        route_key: str,
        recipe_name: str,
        query: str,
        slots: Dict[str, Any],
        top_k: int,
    ) -> List[Document]:
        """具名实体的属性查询：先绑实体，再取属性。"""
        # 新增 graph_entity 路由时，这里要同步加分支；
        # 测试会检查所有 graph_entity 路由都有处理结果。
        if route_key == "recipe_ingredients":
            return self.graph.ingredients_docs(recipe_name, query)
        if route_key == "recipe_steps_time":
            return self.graph.steps_docs(recipe_name, query)
        if route_key == "recipe_overview":
            return self.graph.overview_docs(recipe_name, query)
        if route_key == "recommend_list":
            return self.graph.candidate_docs(slots, top_k)
        if route_key == "nutrition_value":
            return self.graph.value_docs(recipe_name, query)
        if route_key == "nutrition_benefit":
            docs = self.graph.benefit_docs(recipe_name, query)
            docs.extend(self.text.recipe_search(recipe_name, query, top_k))
            return docs
        return []

    def _graph_relation_docs(self, query: str, top_k: int) -> List[Document]:
        """通用图关系检索；走现有 graph_rag_search，不改它的签名。"""
        if self.graph_retriever is None or not hasattr(self.graph_retriever, "graph_rag_search"):
            return []
        try:
            return self.graph_retriever.graph_rag_search(query, top_k)
        except Exception as exc:
            logger.warning("graph_rag_search 失败: %s", exc)
            return []

    # ------------------------------------------------------------------
    # tips 证据
    # ------------------------------------------------------------------

    def _pairing_docs(self, query: str) -> List[Document]:
        entries = self.tips.lookup_pairings(query)
        return [
            _document(entry.to_text(), {
                "source": "tips:pairing",
                "evidence_type": "pairing",
                "pair_found": True,
                "pair_key": entry.pair_key,
                "disclaimer_required": True,
            })
            for entry in entries
        ]

    def _safety_docs(self, query: str) -> List[Document]:
        entry = self.tips.lookup_safety(query)
        if entry is None:
            return []
        return [_document(entry.to_text(), {
            "source": "tips:safety",
            "evidence_type": "safety",
            "section_found": True,
            "section": entry.section,
            "subgroup": entry.subgroup,
            "disclaimer_required": True,
        })]

    def _technique_docs(self, query: str) -> List[Document]:
        entry = self.tips.lookup_technique(query)
        if entry is None:
            return []
        return [_document(entry.to_text(), {
            "source": "tips:technique",
            "evidence_type": "technique",
        })]

    def _scenario_docs(self, query: str, slots: Dict[str, Any]) -> List[Document]:
        plan = self.tips.meal_plan_from_text(query, slots)
        if plan is None:
            return []

        docs = [_document(plan.to_text(), {
            "source": "tips:meal_plan",
            "evidence_type": "meal_plan",
            "rule_applied": True,
            "servings": plan.people,
            "dishes": plan.dishes,
            "meat": plan.meat,
            "veg": plan.veg,
            "add_fish": plan.add_fish,
        })]
        docs.extend(self.graph.meal_candidates_docs(plan, 10))
        return docs

    # ------------------------------------------------------------------
    # 组合检索
    # ------------------------------------------------------------------

    def combined_search(self, query: str, top_k: int) -> List[Document]:
        """传统 + 图并行，round-robin 合并；保留旧路由器的兜底能力。"""
        traditional_k = max(1, top_k // 2)
        graph_k = max(1, top_k - traditional_k)

        traditional_docs: List[Document] = []
        graph_docs: List[Document] = []

        def run_traditional():
            nonlocal traditional_docs
            traditional_docs = self.text.search(query, traditional_k)

        def run_graph():
            nonlocal graph_docs
            graph_docs = self._graph_relation_docs(query, graph_k)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_traditional), executor.submit(run_graph)]
            concurrent.futures.wait(futures, timeout=30)

        merged: List[Document] = []
        seen = set()
        for index in range(max(len(traditional_docs), len(graph_docs))):
            for doc, source in (
                (graph_docs[index] if index < len(graph_docs) else None, "graph_rag"),
                (traditional_docs[index] if index < len(traditional_docs) else None, "traditional"),
            ):
                if doc is None:
                    continue
                key = (doc.metadata.get("node_id"), doc.page_content[:80])
                if key in seen:
                    continue
                seen.add(key)
                doc.metadata["search_source"] = source
                merged.append(doc)

        return merged[:top_k]

    @staticmethod
    def _recipe_name(intent) -> str:
        """取意图里的菜谱实体；菜系/食材不能当菜名用。"""
        names = intent.entity_names("recipe")
        if names:
            return names[0]
        unknown = [entity.name for entity in intent.entities if entity.type == "unknown"]
        return unknown[0] if unknown else ""
