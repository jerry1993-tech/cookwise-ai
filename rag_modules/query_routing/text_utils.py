"""
纯文本工具：实体词典匹配、规则兜底关键词、多意图切分。

实体名来自图谱，优先用词典匹配；兜底关键词需要切中文时用 jieba，
jieba 不可用再退回标点分块。
"""

import logging
import re
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import jieba

    jieba.setLogLevel(logging.WARNING)
except ImportError:  # 允许未安装 jieba 时降级，不阻塞路由
    jieba = None

# 只保留含中文、字母或数字的 token；纯标点不参与关键词
WORD_TOKEN = re.compile(r"[一-龥A-Za-z0-9]")


def normalize_text(text: str) -> str:
    """去掉首尾空白和常见包裹符号，保留原文其余内容。"""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r'^[\s"\'「」『』《》【】\[\]（）()]+', "", text)
    text = re.sub(r'[\s"\'「」『』《》【】\[\]（）()]+$', "", text)
    return text.strip()


def dedupe(items: Iterable[str]) -> List[str]:
    """保序去重。"""
    result: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def contains_any(text: str, terms: Sequence[str]) -> bool:
    """text 中是否出现任意一个词。"""
    return any(term and term in text for term in terms)


def as_name_list(items: Iterable) -> List[str]:
    """把 GraphNode / 字符串混合列表统一成名字列表。"""
    names: List[str] = []
    for item in items or []:
        name = getattr(item, "name", item)
        if name:
            names.append(str(name))
    return names


def extract_known_entities(
    query: str,
    recipes: Sequence[str] = (),
    ingredients: Sequence[str] = (),
    cuisines: Sequence[str] = (),
    aliases: Dict[str, Tuple[str, ...]] = None,
    generic_names: Iterable[str] = (),
) -> List[Tuple[str, str]]:
    """
    从查询里抽取图谱/文档中真实存在的实体，返回 [(标准名, 类型), ...]。

    规则：
    - 长词优先，已被更长实体覆盖的短词不再重复加入；
    - 食材名长度 < 2 不参与（避免“盐”“油”命中几乎所有句子）；
    - generic_names 里的占位词不参与，例如“菜谱”“食材”；
    - 别名命中时映射回标准名，例如“全脂牛奶”->“牛奶”。
    """
    aliases = aliases or {}
    generic_names = set(generic_names)
    candidates: List[Tuple[str, str]] = []

    for name in recipes:
        name = str(name)
        if name and name not in generic_names:
            candidates.append((name, "recipe"))
    for name in ingredients:
        name = str(name)
        if len(name) >= 2 and name not in generic_names:
            candidates.append((name, "ingredient"))
    for name in cuisines:
        name = str(name)
        if name and name not in generic_names:
            candidates.append((name, "cuisine"))

    # 别名：用户说“全脂牛奶”时，补一个标准名“牛奶”
    for canonical, variants in aliases.items():
        for variant in variants:
            if variant != canonical and len(variant) >= 2:
                candidates.append((canonical, "ingredient"))

    # 长词优先，避免“牛奶”把“全脂牛奶”拆成两个
    candidates.sort(key=lambda item: len(item[0]), reverse=True)

    matched: List[Tuple[str, str]] = []
    matched_names: List[str] = []
    for name, entity_type in candidates:
        if not name or name not in query:
            continue
        if any(name in longer for longer in matched_names):
            continue
        if any(existing[0] == name for existing in matched):
            continue
        matched.append((name, entity_type))
        matched_names.append(name)

    # 按实体在 query 中出现的先后排序，保证“宫保鸡丁和可乐鸡翅”不会变成字母序
    def position(item: Tuple[str, str]) -> int:
        name = item[0]
        found = [query.find(name)]
        for variant in aliases.get(name, ()):
            found.append(query.find(variant))
        valid = [pos for pos in found if pos >= 0]
        return min(valid) if valid else len(query) + 1

    matched.sort(key=position)
    return matched


def segment_words(text: str) -> List[str]:
    """把中文文本切成词；优先 jieba，未安装时退回标点分块。"""
    text = str(text or "").strip()
    if not text:
        return []
    if jieba is not None:
        words = jieba.lcut(text)
    else:
        words = re.split(r"[，,。.！!？?；;、\s]+", text)
    words = [str(word).strip() for word in words]
    return [word for word in words if word and WORD_TOKEN.search(word)]


def fallback_keywords(
    query: str,
    recipes: Sequence[str] = (),
    ingredients: Sequence[str] = (),
    cuisines: Sequence[str] = (),
    aliases: Dict[str, Tuple[str, ...]] = None,
    topic_terms: Iterable[str] = (),
    generic_names: Iterable[str] = (),
) -> Tuple[List[str], List[str]]:
    """
    LLM 关键词抽取失败时的兜底，返回 (实体级关键词, 主题级关键词)。

    比原来的 ``query.split()`` 好在：中文按 jieba 切成词，
    实体名直接来自图谱词表，BM25/图索引能真正命中。
    """
    query = normalize_text(query)
    if not query:
        return [], []

    known = extract_known_entities(
        query, recipes, ingredients, cuisines, aliases, generic_names
    )
    entity_keywords = [name for name, _ in known]

    topics = [term for term in topic_terms if term and term in query]
    # 命中了“不辣”就不再保留被它包含的“辣”，避免主题词互相打架
    topics = [
        term for term in topics
        if not any(term != other and term in other for other in topics)
    ]

    # 去掉实体名后，用 jieba 把剩余文本切成主题词
    remainder = query
    for name in sorted(entity_keywords, key=len, reverse=True):
        remainder = remainder.replace(name, " ")

    chunks = segment_words(remainder)
    stopwords = (
        "的", "了", "吗", "呢", "呀", "吧", "啊", "请问", "帮我", "我想",
        "需要", "哪些", "什么", "怎么", "如何", "可以", "推荐", "一下",
        "几个", "几道", "一道", "一些", "准备", "适合", "做法", "有哪些",
    )
    for chunk in chunks:
        chunk = chunk.strip()
        for word in stopwords:
            chunk = chunk.replace(word, "")
        chunk = chunk.strip()
        # 跳过“辣”之于“不辣”这类互相包含的碎片，避免语义冲突
        if any(chunk in topic or topic in chunk for topic in topics):
            continue
        if 2 <= len(chunk) <= 12:
            topics.append(chunk)

    if not entity_keywords and not topics:
        topics = [query[:20]]

    return dedupe(entity_keywords)[:3], dedupe(topics)[:3]


def split_segments(query: str) -> List[str]:
    """
    按问号和并列连词切子问题。

    只切“再/还有/同时/另外/并且”，不按逗号切：
    “8个人在家聚餐，荤素怎么配”里的逗号前后是同一个场景，切开反而丢约束。
    """
    query = normalize_text(query)
    if not query:
        return []

    parts = re.split(r"[？?；;]|(?:再|还有|同时|另外|并且)", query)
    segments = [part.strip(" ，,。.") for part in parts]
    segments = [part for part in segments if len(part) >= 2]
    return segments or [query]


def has_reference(text: str, reference_terms: Sequence[str]) -> bool:
    """是否包含“它/这个/再来一个/多少盐”这类需要指代消解的词。"""
    return contains_any(text, reference_terms)


def clamp_float(value, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
    """把任意输入钳制成 [low, high] 的浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def safe_int(value, default: int = 0) -> int:
    """安全转 int；失败返回默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
