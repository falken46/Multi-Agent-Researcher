"""Writer Agent 节点实现。"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from agents.prompt_loader import load_prompt
from agents.state import ResearchState

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class WriterError(RuntimeError):
    """Writer Agent 写作失败时抛出。"""


def writer_node(state: ResearchState) -> dict[str, object]:
    """根据研究资料生成 Markdown 报告。"""
    topic = state.get("topic", "").strip()
    sub_questions = state.get("sub_questions", [])
    research_results = state.get("research_results", {})
    errors = list(state.get("errors", []))
    logger.info(
        "writer_node enter topic=%r sub_questions=%s results=%s errors=%s",
        topic[:100],
        len(sub_questions),
        len(research_results),
        len(errors),
    )

    try:
        if not topic:
            raise WriterError("topic must not be empty")
        if not sub_questions:
            raise WriterError("sub_questions must not be empty")
        if not research_results:
            raise WriterError("research_results must not be empty")

        system_prompt = load_prompt("writer_system")
        user_prompt = build_writer_prompt(
            topic=topic,
            sub_questions=sub_questions,
            research_results=research_results,
            errors=errors,
        )
        final_report = _call_writer_model(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        logger.info("writer_node output final_report_chars=%s", len(final_report))
        return {"final_report": final_report}
    except Exception as exc:
        logger.error("writer_node failed: %s", exc)
        errors.append(f"Writer: {exc}")
        return {"errors": errors}


def build_writer_prompt(
    topic: str,
    sub_questions: list[str],
    research_results: dict[str, str],
    errors: list[str] | None = None,
) -> str:
    """拼装 Writer Agent 的用户输入 prompt。"""
    question_blocks = []
    for index, question in enumerate(sub_questions, start=1):
        result = research_results.get(question, "资料不足或未检索到结果。")
        question_blocks.append(
            "\n".join(
                [
                    f"### 子问题 {index}",
                    f"问题: {question}",
                    "资料摘要:",
                    result,
                ]
            )
        )

    error_block = "\n".join(f"- {error}" for error in errors or [])
    if not error_block:
        error_block = "无"

    return "\n\n".join(
        [
            f"研究主题:\n{topic}",
            "子问题与资料:",
            "\n\n".join(question_blocks),
            "流程错误记录:",
            error_block,
            "请基于以上资料生成完整 Markdown 研究报告。",
        ]
    )


def _call_writer_model(user_prompt: str, system_prompt: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise WriterError("DEEPSEEK_API_KEY is not configured")
    _validate_ascii_env_value("DEEPSEEK_API_KEY", api_key)

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip(),
    )
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise WriterError("writer model returned empty content")
    return content.strip()


def _validate_ascii_env_value(name: str, value: str) -> None:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise WriterError(f"{name} must contain ASCII characters only") from exc


__all__ = ["build_writer_prompt", "writer_node"]
