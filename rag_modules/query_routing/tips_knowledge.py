"""
CookWise 的 tips 知识库：相克、食品安全、配菜规则、技法。

这些内容目前没有进 Milvus，也不需要为了 10 条相克数据重建索引；
启动时直接解析 Markdown 成结构化对象，查不到就走“无数据”兜底。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .domain import ALIASES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class PairingEntry:
    """一条相克/同食条目。"""

    left: str
    right: str
    title: str
    principle: str
    impact: str
    advice: str
    note: str = ""
    source: str = "data/tips/食材相克与禁忌.md"

    @property
    def pair_key(self) -> str:
        return f"{self.left}+{self.right}"

    def to_text(self) -> str:
        parts = [
            f"【食材搭配】{self.left} + {self.right}",
            f"【相克原理】{self.principle}",
            f"【可能影响】{self.impact}",
            f"【健康建议】{self.advice}",
        ]
        if self.note:
            parts.append(f"【提示】{self.note}")
        parts.append(f"【来源】{self.source}")
        return "\n".join(parts)


@dataclass
class SafetyEntry:
    """食品安全文档里的一个段落/人群分组。"""

    section: str
    subgroup: str
    items: List[str] = field(default_factory=list)
    text: str = ""
    source: str = "data/tips/learn/食品安全.md"

    def to_text(self) -> str:
        title = self.section + (f"（{self.subgroup}）" if self.subgroup else "")
        lines = [f"【食品安全】{title}"]
        if self.items:
            lines.extend(f"- {item}" for item in self.items)
        elif self.text:
            lines.append(self.text.strip())
        lines.append(f"【来源】{self.source}")
        return "\n".join(lines)


@dataclass
class TechniqueEntry:
    """一条技法文档（去腥、焯水、腌制等）。"""

    title: str
    text: str
    source: str

    def to_text(self) -> str:
        return f"【技法】{self.title}\n{self.text.strip()}\n【来源】{self.source}"


@dataclass
class MealPlan:
    """人数 -> 菜数/荤素/加鱼规则。"""

    people: int
    dishes: int
    meat: int
    veg: int
    add_fish: bool
    add_sweet: bool
    source: str = "data/tips/如何选择现在吃什么.md"

    def to_text(self) -> str:
        fish_text = "建议加鱼类荤菜" if self.add_fish else "不强制加鱼（规则是“超过 8 人”才建议加鱼）"
        sweet_text = "有小孩，可增加一道甜味菜" if self.add_sweet else "无小孩，不需要额外加甜味菜"
        return (
            f"【配菜规则】人数 N={self.people}\n"
            f"- 菜的数量 = 人数 + 1 = {self.dishes}\n"
            f"- 素菜 a = floor((N+1)/2) = {self.veg}\n"
            f"- 荤菜 b = ceil((N+1)/2) = {self.meat}\n"
            f"- 鱼：{fish_text}\n"
            f"- 甜味菜：{sweet_text}\n"
            f"- 荤菜动物顺序：猪、鸡、牛、羊、鸭、鱼；不要全用同一种动物\n"
            f"【来源】{self.source}"
        )


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------


class TipsKnowledgeBase:
    """解析并查询 data/tips 下的 Markdown。"""

    TECHNIQUE_FILES = {
        "去腥": "去腥.md",
        "焯水": "学习焯水.md",
        "腌制": "学习腌.md",
        "腌": "学习腌.md",
        "蒸": "学习蒸.md",
        "煮": "学习煮.md",
        "炒": "学习炒与煎.md",
        "煎": "学习炒与煎.md",
        "凉拌": "学习凉拌.md",
        "微波炉": "微波炉.md",
        "空气炸锅": "空气炸锅.md",
        "高压锅": "高压力锅.md",
    }

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else Path(__file__).resolve().parents[2] / "data"
        self.pairings: List[PairingEntry] = self._parse_pairings()
        self.safety: Dict[str, Dict[str, SafetyEntry]] = self._parse_safety()
        logger.info(
            "tips 知识库加载完成：相克 %d 条，食品安全 %d 段",
            len(self.pairings),
            sum(len(group) for group in self.safety.values()),
        )

    # ------------------------------------------------------------------
    # 相克
    # ------------------------------------------------------------------

    def _parse_pairings(self) -> List[PairingEntry]:
        path = self.data_root / "tips" / "食材相克与禁忌.md"
        text = self._read(path)
        if not text:
            return []

        entries: List[PairingEntry] = []
        blocks = re.split(r"(?=^\d+\.\s+\*\*)", text, flags=re.MULTILINE)
        for block in blocks:
            title_match = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", block, flags=re.DOTALL)
            if not title_match:
                continue

            title = title_match.group(1).strip()
            head = re.split(r"[：:]", title, maxsplit=1)[0]
            parts = [part.strip() for part in head.split("+") if part.strip()]
            if len(parts) < 2:
                continue

            note = ""
            if "被夸大" in block or "无需过度恐慌" in block:
                note = "原文标注该说法被夸大，日常适量食用无需过度恐慌。"

            entries.append(PairingEntry(
                left=parts[0],
                right=parts[1],
                title=title,
                principle=self._extract_labeled(block, "相克原理"),
                impact=self._extract_labeled(block, "可能影响"),
                advice=self._extract_labeled(block, "健康建议"),
                note=note,
            ))
        return entries

    @staticmethod
    def _extract_labeled(block: str, label: str) -> str:
        """提取 ``**label**：...`` 到下一个 ``**label**`` 之前的内容。"""
        pattern = re.compile(
            r"\*\*" + re.escape(label) + r"\*\*[：:]\s*(.*?)(?=\n\s*\*\s*\*\*|\Z)",
            flags=re.DOTALL,
        )
        match = pattern.search(block)
        if not match:
            return ""
        text = match.group(1)
        text = re.sub(r"\s*\n\s*", " ", text)
        return text.strip().strip("*").strip()

    def lookup_pairings(self, query: str, limit: int = 3) -> List[PairingEntry]:
        """查询相克条目：优先“两个食材都出现”，其次“只出现一个 + 搭配语气”。"""
        query = self._normalize(query)
        if not query:
            return []

        both: List[PairingEntry] = []
        single: List[PairingEntry] = []
        for entry in self.pairings:
            left_hit = self._term_in_query(query, entry.left)
            right_hit = self._term_in_query(query, entry.right)
            if left_hit and right_hit:
                both.append(entry)
            elif left_hit or right_hit:
                single.append(entry)

        if both:
            # 命中的词越长越具体，优先返回
            both.sort(key=lambda item: len(item.left) + len(item.right), reverse=True)
            return both[:limit]

        if any(term in query for term in ("相克", "搭配", "禁忌", "不能和什么", "同食", "一起吃")):
            return single[:limit]
        return []

    def lookup_pairing(self, query: str) -> Optional[PairingEntry]:
        """返回最匹配的一条；没有返回 None。"""
        matches = self.lookup_pairings(query, limit=1)
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # 食品安全
    # ------------------------------------------------------------------

    def _parse_safety(self) -> Dict[str, Dict[str, SafetyEntry]]:
        path = self.data_root / "tips" / "learn" / "食品安全.md"
        text = self._read(path)
        if not text:
            return {}

        result: Dict[str, Dict[str, SafetyEntry]] = {}
        blocks = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
        for block in blocks:
            title_match = re.match(r"^##\s+(.+)", block, flags=re.MULTILINE)
            if not title_match:
                continue
            section = title_match.group(1).strip()
            body = block[title_match.end():]

            if section == "过敏":
                adult = self._parse_allergy_group(body, "成人")
                child = self._parse_allergy_group(body, "儿童")
                combined_items = adult.items + [item for item in child.items if item not in adult.items]
                result[section] = {
                    "成人": adult,
                    "儿童": child,
                    "": SafetyEntry(section=section, subgroup="", items=combined_items, text=body),
                }
                continue

            items = self._extract_bullets(body)
            result[section] = {
                "": SafetyEntry(section=section, subgroup="", items=items, text=body.strip()),
            }
        return result

    def _parse_allergy_group(self, body: str, group: str) -> SafetyEntry:
        """解析“成人：... / 儿童：...”分组。"""
        pattern = re.compile(
            rf"{group}[：:]\s*(.*?)(?=\n\s*(?:成人|儿童)[：:]|\Z)",
            flags=re.DOTALL,
        )
        match = pattern.search(body)
        text = match.group(1) if match else ""
        return SafetyEntry(
            section="过敏",
            subgroup=group,
            items=self._extract_bullets(text),
            text=text.strip(),
        )

    @staticmethod
    def _extract_bullets(text: str) -> List[str]:
        """把 Markdown 列表项抽成纯文本，去掉链接和空项。"""
        items: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith(("* ", "- ")):
                continue
            item = line[2:].strip()
            item = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", item)  # 去掉 Markdown 链接
            item = item.strip("。；; ")
            if item and item not in items and "……" not in item:
                items.append(item)
        return items

    def lookup_safety(self, query: str) -> Optional[SafetyEntry]:
        """按关键词定位食品安全段落；过敏问题再按人群细分。"""
        query = self._normalize(query)
        if not query:
            return None

        section = ""
        if "过敏" in query or "过敏原" in query:
            section = "过敏"
        elif "沙门" in query:
            section = "沙门氏菌感染"
        elif "黄曲" in query or "霉变" in query or "发霉" in query:
            section = "黄曲霉素"
        elif any(term in query for term in ("3-硝基丙酸", "红心甘蔗", "甘蔗", "椰子")):
            section = "3-硝基丙酸"
        elif any(term in query for term in ("寄生虫", "生鱼片", "田螺", "黄鳝", "牛蛙")):
            section = "寄生虫"
        elif any(term in query for term in ("温度", "中心温度", "熟透", "熟没熟", "熟了吗")):
            section = "食品安全温度"
        elif any(term in query for term in ("中毒", "四季豆", "豆浆", "土豆", "蘑菇", "木耳", "黄花菜")):
            section = "中毒"

        groups = self.safety.get(section)
        if not groups:
            return None

        if section == "过敏":
            if any(term in query for term in ("儿童", "小孩", "孩子", "宝宝")):
                return groups.get("儿童")
            if any(term in query for term in ("成人", "大人")):
                return groups.get("成人")
            return groups.get("")

        return groups.get("")

    # ------------------------------------------------------------------
    # 技法
    # ------------------------------------------------------------------

    def lookup_technique(self, query: str) -> Optional[TechniqueEntry]:
        """按关键词找一篇技法文档。"""
        query = self._normalize(query)
        for keyword, filename in self.TECHNIQUE_FILES.items():
            if keyword in query:
                path = self.data_root / "tips" / "learn" / filename
                text = self._read(path)
                if not text:
                    continue
                title_match = re.match(r"^#\s+(.+)", text, flags=re.MULTILINE)
                title = title_match.group(1).strip() if title_match else filename.replace(".md", "")
                return TechniqueEntry(
                    title=title,
                    text=text[:3000],
                    source=str(path.relative_to(self.data_root.parent)),
                )
        return None

    # ------------------------------------------------------------------
    # 配菜规则
    # ------------------------------------------------------------------

    def meal_plan(self, people: int, has_child: bool = False) -> MealPlan:
        """按 tips 原文的公式计算菜数和荤素；严格 >8 才建议加鱼。"""
        people = max(1, int(people))
        dishes = people + 1
        veg = dishes // 2
        meat = dishes - veg
        return MealPlan(
            people=people,
            dishes=dishes,
            meat=meat,
            veg=veg,
            add_fish=people > 8,
            add_sweet=bool(has_child),
        )

    def meal_plan_from_text(self, query: str, slots: Optional[Dict] = None) -> Optional[MealPlan]:
        """从查询/槽位里取人数和是否有小孩。"""
        slots = slots or {}
        people = slots.get("servings")
        if not people:
            match = re.search(r"(\d+)\s*(?:个)?人", query)
            people = int(match.group(1)) if match else None
        if not people:
            return None

        has_child = bool(slots.get("has_child")) or any(
            term in query for term in ("小孩", "儿童", "孩子", "宝宝")
        )
        return self.meal_plan(int(people), has_child=has_child)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("读取 tips 文件失败 %s: %s", path, exc)
            return ""

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", str(text or ""))

    def _term_in_query(self, query: str, term: str) -> bool:
        """检查一个标准词或它的别名是否出现在查询里。"""
        if not term:
            return False

        variants = [term, *ALIASES.get(term, ())]
        # 反向查别名表：条目里写“虾类”，用户说“虾”也能命中
        for canonical, alias_list in ALIASES.items():
            if term in alias_list:
                variants.extend([canonical, *alias_list])

        for variant in variants:
            if variant and variant in query:
                return True

        # “大量维生素C”这类带修饰词的条目，去掉修饰词再匹配
        for prefix in ("大量", "少量", "过多", "适量"):
            if term.startswith(prefix) and term[len(prefix):] in query:
                return True
        return False

    def stats(self) -> Dict[str, int]:
        """给 /api/stats 或日志用的简单计数。"""
        return {
            "pairings": len(self.pairings),
            "safety_sections": sum(len(group) for group in self.safety.values()),
        }
