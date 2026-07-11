"""
鉴权中间件与限流
- 公网接口: 解析客户Token, 校验身份
- 内网接口: IP白名单 + 管理密钥校验
- 全局QPS限流
- 敏感词过滤
"""
import logging
import time
import ipaddress
from typing import Optional, List, Set
from functools import wraps

from fastapi import Request, HTTPException, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from agents.models.schemas import CustomerContext

logger = logging.getLogger(__name__)

# 内网IP白名单段
INTERNAL_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
]

# 管理密钥（从环境变量读取）
ADMIN_SECRET_KEY = ""

# 全局QPS限流
GLOBAL_QPS_LIMIT = 200

# 敏感词列表（示例，生产环境应使用更完善的过滤方案）
SENSITIVE_WORDS: Set[str] = set()


class AuthMiddleware(BaseHTTPMiddleware):
    """
    统一鉴权中间件
    - /api/agent/admin/* -> 内网鉴权
    - /api/agent/* -> 公网客户Token鉴权
    - 其他路径 -> 放行（系统内置路由）
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 系统内置路由放行
        if not path.startswith("/api/agent/"):
            return await call_next(request)

        # 内网管理接口鉴权
        if path.startswith("/api/agent/admin/"):
            if not self._check_internal_access(request):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Forbidden: Internal access only"}
                )
            return await call_next(request)

        # 公网接口鉴权
        if path.startswith("/api/agent/"):
            # WebSocket 在路由层单独鉴权
            if request.scope.get("type") == "websocket":
                return await call_next(request)

            customer_ctx = self._parse_customer_token(request)
            if customer_ctx is None:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized: Invalid or missing customer token"}
                )
            # 将客户上下文注入request.state
            request.state.customer_ctx = customer_ctx

        return await call_next(request)

    def _check_internal_access(self, request: Request) -> bool:
        """内网接口鉴权: IP白名单 + 管理密钥"""
        # 1. IP白名单检查
        client_ip = self._get_client_ip(request)
        if not self._is_internal_ip(client_ip):
            logger.warning("Internal API access from non-internal IP: %s path=%s", client_ip, request.url.path)
            return False

        # 2. 管理密钥校验
        import os
        secret = os.getenv("AGENT_ADMIN_SECRET", "")
        if secret:
            provided_key = request.headers.get("X-Admin-Key", "")
            if provided_key != secret:
                logger.warning("Invalid admin key from %s", client_ip)
                return False

        return True

    def _parse_customer_token(self, request: Request) -> Optional[CustomerContext]:
        """
        解析客户Token
        从 Authorization header 或 query param 中解析
        Token格式: Bearer <customer_id>:<plan>:<industry>
        生产环境应使用JWT等安全方案
        """
        auth_header = request.headers.get("Authorization", "")
        token_param = request.query_params.get("token", "")

        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        elif token_param:
            token = token_param

        if not token:
            return None

        try:
            # 解析Token (简化格式: customer_id:plan:industry)
            parts = token.split(":")
            if len(parts) >= 1:
                customer_id = parts[0]
                plan = parts[1] if len(parts) > 1 else "free"
                industry = parts[2] if len(parts) > 2 else ""

                if not customer_id:
                    return None

                return CustomerContext(
                    customer_id=customer_id,
                    plan=plan,
                    industry=industry,
                )
            return None
        except Exception as e:
            logger.warning("Failed to parse customer token: %s", e)
            return None

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """获取客户端真实IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "0.0.0.0"

    @staticmethod
    def _is_internal_ip(ip_str: str) -> bool:
        """检查是否为内网IP"""
        try:
            ip = ipaddress.ip_address(ip_str)
            return any(ip in network for network in INTERNAL_NETWORKS)
        except ValueError:
            return False


def filter_sensitive_words(text: str) -> str:
    """敏感词过滤"""
    if not SENSITIVE_WORDS:
        return text
    for word in SENSITIVE_WORDS:
        text = text.replace(word, "***")
    return text


def parse_websocket_token(websocket: WebSocket) -> Optional[CustomerContext]:
    """WebSocket连接鉴权"""
    token = websocket.query_params.get("token", "")
    if not token:
        token = websocket.headers.get("Authorization", "").replace("Bearer ", "")

    if not token:
        return None

    try:
        parts = token.split(":")
        if len(parts) >= 1:
            return CustomerContext(
                customer_id=parts[0],
                plan=parts[1] if len(parts) > 1 else "free",
                industry=parts[2] if len(parts) > 2 else "",
            )
        return None
    except Exception:
        return None
