"""
生成集成模块。

新增两类入口：
- generate_adaptive_answer(...)：有证据时按证据回答
- generate_direct_answer(...)：闲聊/域外/澄清时直接回答，不检索

旧调用方式（question, documents）保持不变。
"""

import concurrent.futures
import copy
import logging
import os
import re
import time
from typing import List

from openai import OpenAI
from langchain_core.documents import Document

from .query_routing.prompts import (
    ANSWER_RULES,
    build_direct_answer_prompt,
    build_no_evidence_prompt,
)

logger = logging.getLogger(__name__)

# 一步一行才算“分条”；用于判断步骤回答有没有被模型压成一段
_LIST_LINE = re.compile(r"^\s*(?:\d+[.、)]|[-*+])\s+\S")
_HOWTO_HINTS = (
    "怎么做", "咋做", "如何做", "怎样做", "怎么制作", "如何制作",
    "做法", "制作方法", "制作步骤", "步骤", "流程",
)
_STEP_FORMAT_RULE = "制作过程每一步单独一行并用 1./2. 编号，禁止把多个步骤合并成一段。"

# 多意图每节一次 LLM 调用；最多同时跑 4 节，避免一个请求把线程数打爆
_MAX_SECTION_WORKERS = 4


def _is_howto(question: str) -> bool:
    """问句是否在问制作过程；只有这类问句才强制分条，避免“要煮多久”被展开。"""
    return any(hint in (question or "") for hint in _HOWTO_HINTS)


class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """初始化生成集成模块"""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 统一的 LLM 客户端配置（支持所有兼容 OpenAI 格式的供应商）
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.moonshot.cn/v1")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )

        logger.info(f"生成模块初始化完成，模型: {model_name}, API地址: {self.base_url}")

    def _create_completion(self, **kwargs):
        """统一关闭 deepseek-flash 的思考输出，避免 reasoning 吃光 max_tokens。"""
        try:
            return self.client.chat.completions.create(
                **kwargs,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except TypeError:
            # 兼容不支持 extra_body 的 OpenAI 兼容客户端
            return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _multi_intent_sections(analysis, documents: List[Document]):
        """按子问题分节；返回 [(序号, 子问题, 证据, 子分析)]，无法安全拆分时返回 None。

        编号由调用方按序号写出，模型只负责每节正文。节标题用 markdown 标题块而不是
        `1. …` 列表项：否则渲染时标题会和正文里的有序列表合并，步骤从 2 开始编号。
        子问题有依赖、证据未按子问题打标或某节没有证据时返回 None，继续走原来的单次生成。
        """
        if analysis is None or not getattr(analysis, "is_multi_intent", False):
            return None
        subs = [
            item for item in (getattr(analysis, "sub_queries", None) or [])
            if item.get("query")
        ]
        if len(subs) < 2 or any(item.get("depends_on", -1) >= 0 for item in subs):
            return None

        grouped = {}
        for doc in documents or []:
            index = doc.metadata.get("sub_query_index")
            if index is None:
                return None
            grouped.setdefault(index, []).append(doc)

        sections = []
        for order, item in enumerate(subs, start=1):
            section_documents = grouped.get(item.get("index", order - 1)) or []
            if not section_documents:
                return None
            sub_analysis = copy.copy(analysis)
            sub_analysis.route_key = item.get("route_key", analysis.route_key)
            sub_analysis.is_multi_intent = False
            sub_analysis.sub_queries = []
            # 免责声明只加在最后一节，避免每节重复
            sub_analysis.disclaimer_required = bool(
                getattr(analysis, "disclaimer_required", False)
            ) and order == len(subs)
            sections.append((order, item["query"], section_documents, sub_analysis))
        return sections

    @staticmethod
    def _strip_section_heading(text: str, question: str) -> str:
        """去掉模型复述的子问题标题，避免与代码生成的节标题重复。"""
        first, separator, rest = text.partition("\n")
        echoed = re.sub(r"^[#>*`\s]*\d+[.、)]\s*", "", first).strip(" *#>`")
        if separator and echoed.strip("？?。") == question.strip("？?。"):
            return rest.lstrip("\n")
        return text

    @staticmethod
    def _evidence_steps(documents: List[Document]):
        """从图谱步骤证据里取出 [{title, description, time}]，用于结构化兜底。"""
        steps = []
        for doc in documents:
            if doc.metadata.get("evidence_type") != "steps":
                continue
            current = None
            for line in doc.page_content.splitlines():
                title = re.match(r"^\s*\d+[.、)]\s+(.+)$", line)
                if title:
                    current = {"title": title.group(1).strip(), "description": "", "time": ""}
                    steps.append(current)
                    continue
                if current is None:
                    continue
                field = re.match(r"^\s*(描述|时间)：(.*)$", line)
                if field:
                    key = "description" if field.group(1) == "描述" else "time"
                    current[key] = field.group(2).strip()
        return steps

    @staticmethod
    def _ensure_step_structure(
        question: str, content: str, documents: List[Document], analysis=None
    ) -> str:
        """步骤回答被压成一段时，按证据重新分条——结构不取决于模型是否听话。

        只在问句确实在问“怎么做”时兜底，避免“要煮多久”这类单点问题被展开成整套步骤；
        整篇多意图回答不兜底，否则会把食材等其他子问题的内容替换掉。
        """
        if getattr(analysis, "is_multi_intent", False) or not _is_howto(question):
            return content
        steps = GenerationIntegrationModule._evidence_steps(documents)
        if len(steps) < 3:
            return content
        if sum(1 for line in content.splitlines() if _LIST_LINE.match(line)) >= 2:
            return content

        logger.warning("步骤回答缺少分条，按证据重建结构")
        lines = []
        for index, step in enumerate(steps, start=1):
            text = step["description"] or step["title"]
            if step["time"]:
                text = f"{text}（{step['time']}）"
            lines.append(f"{index}. {text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 多意图分节：有界并行，按序输出
    # ------------------------------------------------------------------

    def _section_tasks(self, sections):
        """把每一节提交到有界线程池，返回 (executor, [(order, question, future)])。"""
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_MAX_SECTION_WORKERS, len(sections)),
            thread_name_prefix="section-gen",
        )
        tasks = []
        for order, sub_question, section_documents, sub_analysis in sections:
            future = executor.submit(
                self.generate_adaptive_answer, sub_question, section_documents, sub_analysis
            )
            tasks.append((order, sub_question, future))
        return executor, tasks

    @staticmethod
    def _section_result(future, sub_question: str) -> str:
        """取单节结果；单节失败只影响这一节，不拖垮整篇回答。"""
        try:
            return future.result()
        except Exception as exc:
            logger.error("子问题生成失败（%s）: %s", sub_question, exc)
            return "抱歉，这个子问题生成失败，请稍后重试。"

    def _generate_multi_intent(self, sections) -> str:
        """非流式：所有节并行生成，按原顺序拼回去。"""
        executor, tasks = self._section_tasks(sections)
        try:
            parts = []
            for order, sub_question, future in tasks:
                content = self._section_result(future, sub_question)
                content = self._strip_section_heading(content, sub_question)
                parts.append(f"### {order}. {sub_question}\n\n{content}".strip())
            return "\n\n".join(parts)
        finally:
            executor.shutdown(wait=True)

    def _stream_one_section(self, question: str, documents: List[Document], analysis):
        """流式生成一节；先缓冲到第一行结束，去掉模型复述的标题再往外吐。"""
        buffer = ""
        emitted = False

        def first_line_ready(text: str) -> str:
            # 模型有时先吐几个空行；跳过它们，否则会把空行当成“第一行”而漏掉标题
            return re.sub(r"^(?:[ \t]*\r?\n)+", "", text)

        for chunk in self.generate_adaptive_answer_stream(question, documents, analysis=analysis):
            if emitted:
                yield chunk
                continue
            buffer += chunk
            ready = first_line_ready(buffer)
            if "\n" not in ready:
                continue
            cleaned = self._strip_section_heading(ready, question)
            emitted = True
            if cleaned:
                yield cleaned
        if not emitted:
            yield self._strip_section_heading(first_line_ready(buffer), question)

    def _stream_sections_in_order(self, sections):
        """所有节并行非流式生成，按顺序输出；给需要步骤修复的 howto 首节用。"""
        executor, tasks = self._section_tasks(sections)
        try:
            for order, sub_question, future in tasks:
                yield f"### {order}. {sub_question}\n\n"
                content = self._section_result(future, sub_question)
                yield self._strip_section_heading(content, sub_question)
                yield "\n\n"
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _stream_multi_intent(self, sections):
        """流式：第 1 节边生成边吐；其余节并行生成，轮到哪节再按序输出。"""
        first_order, first_question, first_documents, first_analysis = sections[0]

        # “怎么做”且证据里有 >=3 步时，首节必须走非流式，才能跑 _ensure_step_structure
        if _is_howto(first_question) and len(self._evidence_steps(first_documents)) >= 3:
            yield from self._stream_sections_in_order(sections)
            return

        rest = sections[1:]

        executor = None
        tasks = []
        if rest:
            # 第 1 节已经在当前线程里生成，线程池最多再补 3 节，总数仍不超过 4
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(max(1, _MAX_SECTION_WORKERS - 1), len(rest)),
                thread_name_prefix="section-gen",
            )
            for order, sub_question, section_documents, sub_analysis in rest:
                tasks.append((
                    order,
                    sub_question,
                    executor.submit(
                        self.generate_adaptive_answer,
                        sub_question,
                        section_documents,
                        sub_analysis,
                    ),
                ))

        try:
            yield f"### {first_order}. {first_question}\n\n"
            yield from self._stream_one_section(first_question, first_documents, first_analysis)
            yield "\n\n"

            for order, sub_question, future in tasks:
                yield f"### {order}. {sub_question}\n\n"
                content = self._section_result(future, sub_question)
                yield self._strip_section_heading(content, sub_question)
                yield "\n\n"
        finally:
            # 客户端断开时取消还没开始的任务；已经在跑的 LLM 调用无法强行取消
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # 有证据的回答
    # ------------------------------------------------------------------

    def generate_adaptive_answer(self, question: str, documents: List[Document], analysis=None) -> str:
        """基于证据生成答案；证据为空时明确说“没有检索到”。"""
        sections = self._multi_intent_sections(analysis, documents)
        if sections is not None:
            return self._generate_multi_intent(sections)

        if not documents:
            prompt = build_no_evidence_prompt(question)
        else:
            context = self._format_context(documents, analysis)
            prompt = self._build_prompt(question, context, analysis)

        try:
            response = self._create_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                logger.warning("生成返回空内容（可能被思考过程占用 token）")
            return self._ensure_step_structure(question, content, documents, analysis)
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(
        self,
        question: str,
        documents: List[Document],
        max_retries: int = 3,
        analysis=None,
    ):
        """流式生成；失败时自动切换非流式。

        多意图回答的节标题与编号由代码统一生成：按子问题逐节调用模型（模型只负责正文），
        逐节 yield。这样无论模型是否遵守格式指令，整篇编号都不会重复。
        """
        sections = self._multi_intent_sections(analysis, documents)
        if sections is not None:
            yield from self._stream_multi_intent(sections)
            return

        if not documents:
            prompt = build_no_evidence_prompt(question)
        else:
            context = self._format_context(documents, analysis)
            prompt = self._build_prompt(question, context, analysis)

        started = False  # 已经吐给前端的内容不能撤回，重试会变成重复正文
        for attempt in range(max_retries):
            try:
                response = self._create_completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60
                )

                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        started = True
                        yield chunk.choices[0].delta.content
                return

            except Exception as e:
                if started:
                    # 已经输出了一部分，再重试会把同样的开头重复一遍
                    logger.error("流式生成中途中断，不再重试: %s", e)
                    yield "\n\n（生成中断，请重试）"
                    return

                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue

                logger.error("流式生成完全失败，尝试非流式后备方案")
                print("⚠️ 流式生成失败，切换到标准模式...")
                try:
                    yield self.generate_adaptive_answer(question, documents, analysis)
                except Exception as fallback_error:
                    logger.error(f"后备生成也失败: {fallback_error}")
                    yield f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                return

    # ------------------------------------------------------------------
    # 不检索的直接回答
    # ------------------------------------------------------------------

    def generate_direct_answer(self, question: str, analysis=None) -> str:
        """闲聊/域外/澄清场景；域外使用确定性拒识，避免模型答跑题。"""
        route_key = getattr(analysis, "route_key", "") if analysis is not None else ""
        if route_key == "out_of_scope":
            return (
                "我是 CookWise 烹饪助手，只能回答菜谱、食材、营养和厨房技巧相关的问题。"
                "你可以问我“水煮鱼需要哪些食材”或“减脂期吃什么”。"
            )

        prompt = build_direct_answer_prompt(question, route_key or "chitchat")
        try:
            response = self._create_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512,
            )
            content = (response.choices[0].message.content or "").strip()
            return content or "我在呢～想聊做饭，还是想找一道菜？"
        except Exception as e:
            logger.warning(f"直接回答失败: {e}")
            return "我在呢～想聊做饭，还是想找一道菜？"

    def generate_direct_answer_stream(self, question: str, analysis=None):
        """直接回答的流式版本；失败时退回非流式。"""
        route_key = getattr(analysis, "route_key", "") if analysis is not None else ""
        if route_key == "out_of_scope":
            yield self.generate_direct_answer(question, analysis)
            return

        prompt = build_direct_answer_prompt(question, route_key or "chitchat")
        try:
            response = self._create_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512,
                stream=True,
                timeout=60,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.warning(f"直接流式回答失败: {e}")
            yield self.generate_direct_answer(question, analysis)

    # ------------------------------------------------------------------
    # Prompt 组装
    # ------------------------------------------------------------------

    def _format_context(self, documents: List[Document], analysis=None) -> str:
        """把证据拼成带来源标签的上下文；多意图时标出证据属于哪个子问题。"""
        is_multi = bool(getattr(analysis, "is_multi_intent", False))
        parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if not content:
                continue
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("search_type")
                or doc.metadata.get("retrieval_level")
                or "evidence"
            )
            sub_query = doc.metadata.get("sub_query") or ""
            if is_multi and sub_query:
                source = f"{source}｜子问题：{sub_query}"
            parts.append(f"[{source}] {content}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str, analysis=None) -> str:
        """按路由键追加约束，减少“拿到 5 条就自由发挥”。"""
        route_key = getattr(analysis, "route_key", "") if analysis is not None else ""
        rules = [
            "只使用上下文中出现的信息；上下文没有的食材、用量、时间、营养数字，不要补充。",
            "如果证据不足以回答，明确说“知识库暂未收录”，不要猜测。",
        ]

        sub_queries = getattr(analysis, "sub_queries", None) or []

        # 多意图按子问题套各自规则；能分节生成时这段只作为无法拆分时的兜底。
        numbered_sub_questions: List[str] = []
        sub_rules: List[str] = []
        if getattr(analysis, "is_multi_intent", False):
            for order, item in enumerate(
                (item for item in sub_queries if item.get("query")), start=1
            ):
                numbered_sub_questions.append(f"{order}. {item['query']}")
                sub_rule = ANSWER_RULES.get(item.get("route_key", ""))
                if sub_rule:
                    if item.get("route_key") == "recipe_steps_time" and _is_howto(item["query"]):
                        sub_rule = f"{sub_rule} {_STEP_FORMAT_RULE}"
                    sub_rules.append(f"子问题 {order}（{item['query']}）：{sub_rule}")
            if numbered_sub_questions:
                rules.append(
                    "用户有多个子问题，必须逐个回答，不能遗漏，按 1./2.… 连续编号，"
                    "不要重新从 1 开始：\n  "
                    + "\n  ".join(numbered_sub_questions)
                )
            if sub_rules:
                rules.append("各子问题的回答要求：\n  " + "\n  ".join(sub_rules))

        answer_rule = ANSWER_RULES.get(route_key)
        if answer_rule and not sub_rules:
            if route_key == "recipe_steps_time" and _is_howto(question):
                answer_rule = f"{answer_rule} {_STEP_FORMAT_RULE}"
            rules.append(answer_rule)

        if any(item.get("route_key") == "out_of_scope" for item in sub_queries):
            rules.append("如果用户同时问了非饮食问题，只回答饮食部分，并礼貌说明其他问题超出范围。")
        if any(item.get("depends_on", -1) >= 0 for item in sub_queries):
            rules.append("子问题之间存在依赖，请先回答被依赖的问题，再回答后续问题。")
        slots = getattr(analysis, "slots", None) or {}
        if isinstance(slots, dict) and slots.get("spicy") == "不辣":
            rules.append("用户不吃辣；如果证据显示菜谱含辣椒/麻辣，请明确提示，并给出不辣替代建议。")
        if getattr(analysis, "disclaimer_required", False):
            rules.append("结尾加一句：以上为一般饮食建议，适量、个体差异，特殊健康状况请咨询医生或营养师。")

        rule_text = "\n".join(f"- {rule}" for rule in rules)

        return f"""
作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

回答要求：
{rule_text}

回答：
""".strip()
