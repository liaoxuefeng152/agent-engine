"""
Agent-Engine 路由注册模块
将 Agent-Engine 的 API 路由注册到 FastAPI 应用
"""
import logging

from fastapi import FastAPI

from agents.routes.public_api import router as public_router
from agents.routes.admin_api import router as admin_router
from agents.middleware.auth import AuthMiddleware

logger = logging.getLogger(__name__)


def register_agent_engine_routes(app: FastAPI):
    """
    注册 Agent-Engine 所有路由到 FastAPI 应用
    应在 app 创建后、启动前调用
    """
    # 添加鉴权中间件
    app.add_middleware(AuthMiddleware)

    # 注册公网路由
    app.include_router(public_router)

    # 注册内网管理路由
    app.include_router(admin_router)

    logger.info("Agent-Engine routes registered: public + admin")


def init_agent_engine():
    """
    初始化 Agent-Engine 服务
    - 预加载配置缓存
    - 初始化向量引擎
    """
    from agents.services.config_cache import config_cache

    try:
        config_cache.preload_all()
        logger.info("Agent-Engine config cache preloaded")
    except Exception as e:
        logger.warning("Agent-Engine config preload skipped: %s", e)

    logger.info("Agent-Engine initialized successfully")
