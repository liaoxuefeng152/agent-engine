"""
通用 HTTP 请求工具

提供安全的 HTTP 请求能力，Agent 可通过此工具调用外部 API，
如查询天气、获取汇率、调用第三方服务等。内置超时保护和响应大小限制。
"""
import json
import logging
from typing import Optional

import httpx
from langchain.tools import tool
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)

# 请求限制
_MAX_TIMEOUT = 30  # 秒
_MAX_RESPONSE_SIZE = 1_000_000  # 1MB
_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}

# 内网地址黑名单（防止 SSRF）
_BLOCKED_HOSTS = [
    "localhost", "127.0.0.1", "0.0.0.0",
    "10.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.", "169.254.",
    "metadata.google.internal", "169.254.169.254",
]


def _validate_url(url: str) -> tuple[bool, str]:
    """
    URL 安全校验核心逻辑（可被多个 tool 复用）

    校验规则：
    1. 必须是 http/https 协议
    2. 不能访问内网地址（防 SSRF）
    """
    url_lower = url.lower()

    if not url_lower.startswith(("http://", "https://")):
        return False, "仅支持 http/https 协议。"

    for blocked in _BLOCKED_HOSTS:
        if blocked in url_lower:
            return False, f"禁止访问内网地址: {blocked}"

    return True, ""


def _do_request(
    method: str,
    url: str,
    headers: dict = None,
    params: dict = None,
    body: dict = None,
    timeout: int = _MAX_TIMEOUT,
) -> dict:
    """
    HTTP 请求核心逻辑（可被多个 tool 复用）
    """
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=body if body else None,
        )

        # 检查响应大小
        content = response.content
        if len(content) > _MAX_RESPONSE_SIZE:
            return {
                "status": "error",
                "message": f"响应体过大 ({len(content)} bytes)，超过限制 {_MAX_RESPONSE_SIZE} bytes",
            }

        # 尝试解析 JSON
        try:
            resp_data = response.json()
            resp_text = json.dumps(resp_data, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, ValueError):
            resp_text = response.text[:5000]

        return {
            "status": "success",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "body": resp_text,
        }


@tool
def http_request(
    url: str,
    method: str = "GET",
    headers: str = "",
    query_params: str = "",
    body: str = "",
) -> str:
    """
    Send an HTTP request to an external API.
    Use this tool to fetch data from external services, call third-party APIs,
    or retrieve information from the web.

    Args:
        url: The target URL (must be http/https, internal network addresses are blocked).
        method: HTTP method: GET, POST, PUT, DELETE, PATCH (default: GET).
        headers: Optional headers as JSON string. Example: {"Authorization": "Bearer xxx", "Content-Type": "application/json"}
        query_params: Optional query parameters as JSON string. Example: {"page": "1", "size": "20"}
        body: Optional request body as JSON string (for POST/PUT/PATCH). Example: {"name": "test", "value": 123}

    Returns:
        Response status code, content type, and body content.
    """
    ctx = request_context.get() or new_context(method="http_request")

    # Validate method
    method = method.upper().strip()
    if method not in _ALLOWED_METHODS:
        return f"不支持的 HTTP 方法: {method}。允许: {', '.join(_ALLOWED_METHODS)}"

    # Validate URL
    is_valid, error_msg = _validate_url(url)
    if not is_valid:
        return f"URL 校验失败: {error_msg}"

    # Parse JSON parameters
    parsed_headers = {}
    if headers:
        try:
            parsed_headers = json.loads(headers)
        except json.JSONDecodeError:
            return "headers 参数格式错误，需要合法的 JSON 字符串。"

    parsed_params = {}
    if query_params:
        try:
            parsed_params = json.loads(query_params)
        except json.JSONDecodeError:
            return "query_params 参数格式错误，需要合法的 JSON 字符串。"

    parsed_body = None
    if body:
        try:
            parsed_body = json.loads(body)
        except json.JSONDecodeError:
            return "body 参数格式错误，需要合法的 JSON 字符串。"

    # Execute request
    try:
        result = _do_request(
            method=method,
            url=url,
            headers=parsed_headers,
            params=parsed_params,
            body=parsed_body,
        )

        if result["status"] == "error":
            return f"请求失败: {result['message']}"

        status_code = result["status_code"]
        content_type = result["content_type"]
        resp_body = result["body"]

        output = f"状态码: {status_code}\nContent-Type: {content_type}\n\n响应内容:\n{resp_body}"

        if status_code >= 400:
            output = f"⚠️ 请求返回错误状态码 {status_code}\n\n{output}"

        return output

    except httpx.TimeoutException:
        return f"请求超时（{_MAX_TIMEOUT}秒），请稍后重试或检查目标服务是否可用。"
    except httpx.ConnectError as e:
        return f"连接失败: 无法连接到目标服务器，请检查 URL 是否正确。"
    except Exception as e:
        logger.error(f"HTTP request failed: {e}")
        return f"请求异常: {e}"
