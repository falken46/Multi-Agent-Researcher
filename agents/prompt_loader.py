"""Prompt 文件加载工具。"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptLoadError(RuntimeError):
    """Prompt 文件缺失或无法读取时抛出。"""


def load_prompt(name: str) -> str:
    """按名称加载 prompts 目录下的 Markdown prompt。"""
    safe_name = name.strip()
    if not safe_name:
        raise PromptLoadError("prompt name must not be empty")

    prompt_path = PROMPTS_DIR / f"{safe_name}.md"
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise PromptLoadError(f"prompt file not found: {prompt_path}") from exc

    if not prompt:
        raise PromptLoadError(f"prompt file is empty: {prompt_path}")
    return prompt
