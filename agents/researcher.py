"""Researcher Agent 节点实现。"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from agents.prompt_loader import load_prompt
from agents.state import ResearchState
from tools.web_search import SearchResult, web_search

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
MAX_SEARCH_RESULTS = 3


class ResearcherError(RuntimeError):
    """Researcher Agent 研究失败时抛出。"""


def researcher_node(state: ResearchState) -> dict[str, object]:
    """遍历研究子问题,返回每个子问题对应的资料摘要。"""
    sub_questions = state.get("sub_questions", [])
    logger.info("researcher_node enter sub_questions=%s", len(sub_questions))

    research_results: dict[str, str] = dict(state.get("research_results", {}))
    errors = list(state.get("errors", []))

    if not sub_questions:
        errors.append("Researcher: sub_questions must not be empty")
        return {"errors": errors}

    system_prompt = load_prompt("researcher_system")
    for question in sub_questions:
        try:
            summary = _research_question(question=question, system_prompt=system_prompt)
            research_results[question] = summary
            logger.info("researcher_node question done chars=%s", len(summary))
        except Exception as exc:
            logger.error("researcher_node question failed: %s", exc)
            errors.append(f"Researcher: {question} | {exc}")

    logger.info(
        "researcher_node output research_results=%s errors=%s",
        len(research_results),
        len(errors),
    )
    return {"research_results": research_results, "errors": errors}


def _research_question(question: str, system_prompt: str) -> str:
    normalized_question = question.strip()
    if not normalized_question:
        raise ResearcherError("question must not be empty")

    search_results = web_search(normalized_question, max_results=MAX_SEARCH_RESULTS)
    summary = _call_summary_model(
        question=normalized_question,
        search_results=search_results,
        system_prompt=system_prompt,
    )
    return _append_sources(summary=summary, search_results=search_results)


def _call_summary_model(
    question: str,
    search_results: list[SearchResult],
    system_prompt: str,
) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ResearcherError("DEEPSEEK_API_KEY is not configured")
    _validate_ascii_env_value("DEEPSEEK_API_KEY", api_key)

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip(),
    )
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_summary_prompt(question, search_results)},
        ],
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise ResearcherError("summary model returned empty content")
    return content.strip()


def _build_summary_prompt(question: str, search_results: list[SearchResult]) -> str:
    result_blocks = []
    for index, result in enumerate(search_results, start=1):
        result_blocks.append(
            "\n".join(
                [
                    f"{index}. {result['title']}",
                    f"URL: {result['url']}",
                    f"摘要: {result['snippet']}",
                ]
            )
        )

    return "\n\n".join(
        [
            f"子问题:\n{question}",
            "搜索资料:",
            "\n\n".join(result_blocks),
            "请基于以上资料生成资料摘要,并保留来源 URL。",
        ]
    )


def _append_sources(summary: str, search_results: list[SearchResult]) -> str:
    urls = []
    seen_urls = set()
    for result in search_results:
        url = result["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        urls.append(url)

    source_lines = "\n".join(f"- {url}" for url in urls)
    if "来源:" in summary:
        return summary
    return f"{summary}\n\n来源:\n{source_lines}"


def _validate_ascii_env_value(name: str, value: str) -> None:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ResearcherError(f"{name} must contain ASCII characters only") from exc


__all__ = ["researcher_node"]
