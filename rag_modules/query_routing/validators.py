"""
证据覆盖校验（CRAG 的轻量版）。

原则：能用确定性计数判断的，就不让 LLM 判断。
校验规则按 RouteKeySpec.validator 的名字分发，代码里不写死路由键。
"""

import logging
from typing import Callable, Dict, List

from langchain_core.documents import Document

from .schemas import SubPlan, ValidationResult

logger = logging.getLogger(__name__)


class EvidenceValidator:
    """按 RouteKeySpec.validator 选择校验规则。"""

    def validate(self, sub_plan: SubPlan, documents: List[Document]) -> ValidationResult:
        """返回 complete / partial / missing。"""
        # 不需要检索的路由（chitchat / out_of_scope）配置的 engine 就是 none
        if sub_plan.engine == "none":
            return ValidationResult("complete", "不需要检索")

        if not documents:
            return ValidationResult(
                "missing",
                "未检索到证据",
                retry_hint="text" if sub_plan.engine == "graph_entity" else "",
            )

        handler = VALIDATORS.get(sub_plan.validator)
        if handler is None:
            # 新领域还没注册校验器时，至少有证据就放行，避免整条链路失败
            return ValidationResult("complete", "已返回证据")
        return handler(documents)

    # ------------------------------------------------------------------
    # 各校验规则
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ingredients(documents: List[Document]) -> ValidationResult:
        for doc in documents:
            if doc.metadata.get("evidence_type") != "ingredients":
                continue
            expected = int(doc.metadata.get("expected_count") or 0)
            found = int(doc.metadata.get("found_count") or 0)
            if expected > 0 and found >= expected:
                return ValidationResult("complete", f"图谱食材 {found}/{expected}")
            return ValidationResult(
                "partial",
                f"图谱食材 {found}/{expected}",
                retry_hint="text",
            )

        return ValidationResult(
            "partial",
            "只命中菜谱原文，未命中图谱全量清单",
            retry_hint="text",
        )

    @staticmethod
    def _validate_steps(documents: List[Document]) -> ValidationResult:
        for doc in documents:
            if doc.metadata.get("evidence_type") != "steps":
                continue
            expected = int(doc.metadata.get("expected_count") or 0)
            found = int(doc.metadata.get("found_count") or 0)
            if expected > 0 and found >= expected:
                return ValidationResult("complete", f"步骤 {found}/{expected}")
            return ValidationResult("partial", f"步骤 {found}/{expected}")
        return ValidationResult("partial", "只命中原文步骤")

    @staticmethod
    def _validate_overview(documents: List[Document]) -> ValidationResult:
        return EvidenceValidator._validate_presence(documents, "overview", "菜谱概览")

    @staticmethod
    def _validate_pair(documents: List[Document]) -> ValidationResult:
        return EvidenceValidator._validate_flag(documents, "pair_found", "相克条目")

    @staticmethod
    def _validate_technique(documents: List[Document]) -> ValidationResult:
        return EvidenceValidator._validate_presence(documents, "technique", "技法说明")

    @staticmethod
    def _validate_recommend(documents: List[Document]) -> ValidationResult:
        for doc in documents:
            if doc.metadata.get("evidence_type") != "candidates":
                continue
            count = int(doc.metadata.get("candidate_count") or 0)
            if count >= 3:
                return ValidationResult("complete", f"候选 {count} 道")
            return ValidationResult("partial", f"候选只有 {count} 道")
        return ValidationResult("partial", "只命中文本证据，未命中结构化候选")

    @staticmethod
    def _validate_scenario(documents: List[Document]) -> ValidationResult:
        return EvidenceValidator._validate_flag(documents, "rule_applied", "配菜规则")

    @staticmethod
    def _validate_value(documents: List[Document]) -> ValidationResult:
        return EvidenceValidator._validate_flag(
            documents,
            "value_found",
            "原文数值",
            missing_detail="菜谱原文未标注该数值",
        )

    @staticmethod
    def _validate_benefit(documents: List[Document]) -> ValidationResult:
        return EvidenceValidator._validate_presence(documents, "benefit", "营养描述")

    @staticmethod
    def _validate_safety(documents: List[Document]) -> ValidationResult:
        return EvidenceValidator._validate_flag(documents, "section_found", "食品安全段落")

    @staticmethod
    def _validate_presence(documents: List[Document], evidence_type: str, label: str) -> ValidationResult:
        if any(doc.metadata.get("evidence_type") == evidence_type for doc in documents):
            return ValidationResult("complete", f"命中{label}")
        return ValidationResult("partial", f"只命中辅助证据，未命中{label}")

    @staticmethod
    def _validate_flag(
        documents: List[Document],
        flag: str,
        label: str,
        missing_detail: str = "",
    ) -> ValidationResult:
        if any(doc.metadata.get(flag) for doc in documents):
            return ValidationResult("complete", f"命中{label}")
        return ValidationResult("missing", missing_detail or f"未命中{label}")


# 校验器注册表：键就是 RouteKeySpec.validator；新增领域时在这里注册
VALIDATORS: Dict[str, Callable[[List[Document]], ValidationResult]] = {
    "list": EvidenceValidator._validate_ingredients,
    "steps": EvidenceValidator._validate_steps,
    "overview": EvidenceValidator._validate_overview,
    "pair": EvidenceValidator._validate_pair,
    "technique": EvidenceValidator._validate_technique,
    "recommend": EvidenceValidator._validate_recommend,
    "scenario": EvidenceValidator._validate_scenario,
    "value": EvidenceValidator._validate_value,
    "benefit": EvidenceValidator._validate_benefit,
    "safety": EvidenceValidator._validate_safety,
}
