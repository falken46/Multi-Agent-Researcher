"""Agent 的兼容导入层；实际 Prompt 加载逻辑位于 core。"""

from core.prompts import PromptLoadError, load_prompt

__all__ = ["PromptLoadError", "load_prompt"]
