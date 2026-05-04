"""Researcher Agent 使用的网页抓取工具。"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_CHARS = 5_000
USER_AGENT = (
    "Mozilla/5.0 (compatible; MultiAgentResearchAssistant/0.1; "
    "+https://example.local)"
)


class WebFetchError(RuntimeError):
    """网页无法抓取或解析时抛出。"""


def web_fetch(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """抓取网页,返回清理后的纯文本。"""
    normalized_url = url.strip()
    _validate_url(normalized_url)
    if timeout <= 0:
        raise WebFetchError("timeout must be greater than 0")
    if max_chars < 1:
        raise WebFetchError("max_chars must be greater than 0")

    logger.info(
        "web_fetch input url=%s timeout=%s max_chars=%s",
        normalized_url,
        timeout,
        max_chars,
    )

    try:
        response = requests.get(
            normalized_url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebFetchError(f"failed to fetch {normalized_url}: {exc}") from exc

    text = _extract_text(response.text, max_chars=max_chars)
    logger.info("web_fetch output url=%s chars=%s", normalized_url, len(text))
    return text


def _validate_url(url: str) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise WebFetchError("url must be an absolute http or https URL")


def _extract_text(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    if not text:
        raise WebFetchError("fetched page did not contain readable text")
    return text[:max_chars]
