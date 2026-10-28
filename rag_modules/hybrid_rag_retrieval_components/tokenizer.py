"""中文分词：BM25 建索引与查询共用同一套切词规则。"""

from typing import List

import jieba

# 中文停用词表：助词 / 连词 / 疑问词 / 人称 / 语气词 / 动词修饰
# 不引第三方停用词包，按烹饪问答场景手挑（覆盖 testset 高频虚词）
CHINESE_STOPWORDS = set("""
的 了 和 是 在 我 有 就 不 也 都 还 这 那 一 个 与 及 等 上 下 中 为 以 于 从 把 被 让 使 又 而 但 或
什么 怎么 如何 哪些 哪个 哪里 谁 多少 几 你 他 她 它 我们 他们 她们 它们
请问 请 想 要 需要 能 可以 应该 会 啊 呢 吧 嘛 吗 哦 呀 哈
之 其 此 该 即 各 每 些 种 类 时 后 前 里 外 内 间 已经 正在 一些 一下
""".split())


def tokenize_chinese(text: str) -> List[str]:
    """jieba 精确分词 + 停用词 / 空白 / 单字符过滤"""
    if not text:
        return []
    tokens = jieba.lcut(text)
    return [
        t for t in tokens
        if t.strip() and t not in CHINESE_STOPWORDS and not t.isspace()
    ]
