"""Planner Agent 节点实现。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agents.prompt_loader import load_prompt
from agents.state import ResearchState

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class PlannerError(RuntimeError):
    """Planner Agent 规划失败时抛出。"""


def planner_node(state: ResearchState) -> dict[str, list[str]]:
    """接收研究主题,返回 3-5 个研究子问题。"""
    topic = state.get("topic", "").strip()
    logger.info("planner_node enter topic=%r", topic[:100])

    try:
        if not topic:
            raise PlannerError("topic must not be empty")

        system_prompt = load_prompt("planner_system")
        raw_content = _call_planner_model(topic=topic, system_prompt=system_prompt)
        sub_questions = _parse_sub_questions(raw_content)
        logger.info("planner_node output sub_questions=%s", len(sub_questions))
        return {"sub_questions": sub_questions}
    except Exception as exc:
        logger.error("planner_node failed: %s", exc)
        errors = list(state.get("errors", []))
        errors.append(f"Planner: {exc}")
        return {"errors": errors}


def _call_planner_model(topic: str, system_prompt: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise PlannerError("DEEPSEEK_API_KEY is not configured")
    _validate_ascii_env_value("DEEPSEEK_API_KEY", api_key)

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip(),
    )
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topic},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise PlannerError("planner model returned empty content")
    return content


def _validate_ascii_env_value(name: str, value: str) -> None:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PlannerError(f"{name} must contain ASCII characters only") from exc


def _parse_sub_questions(content: str) -> list[str]:
    payload = json.loads(_strip_json_fence(content))
    raw_questions: Any
    if isinstance(payload, list):
        raw_questions = payload
    elif isinstance(payload, dict):
        raw_questions = payload.get("sub_questions")
    else:
        raise PlannerError("planner output must be a JSON object or array")

    if not isinstance(raw_questions, list):
        raise PlannerError("planner output must contain sub_questions list")

    sub_questions = _normalize_questions(raw_questions)
    if len(sub_questions) < 3:
        raise PlannerError("planner output must contain at least 3 sub questions")
    return sub_questions[:5]


def _strip_json_fence(content: str) -> str:
    stripped_content = content.strip()
    if not stripped_content.startswith("```"):
        return stripped_content

    lines = stripped_content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_questions(raw_questions: list[Any]) -> list[str]:
    sub_questions: list[str] = []
    seen_questions: set[str] = set()

    for raw_question in raw_questions:
        if not isinstance(raw_question, str):
            continue
        question = raw_question.strip()
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        sub_questions.append(question)

    if not sub_questions:
        raise PlannerError("planner output did not contain usable sub questions")
    return sub_questions


__all__ = ["planner_node"]
