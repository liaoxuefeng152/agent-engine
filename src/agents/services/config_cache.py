"""
配置缓存服务 - Agent配置热更新 + 内存缓存
支持后台修改配置后实时热生效，无需重启服务
"""
import logging
import threading
import time
from typing import Optional, Dict
from sqlalchemy import select

from agents.models.agent_models import LarkAgentInfo
from storage.database.db import get_session

logger = logging.getLogger(__name__)


class ConfigCache:
    """
    智能体配置内存缓存（单例）
    - 读取时优先从内存缓存获取
    - 缓存未命中则查数据库并回填
    - 支持主动刷新（管理接口触发）
    - 缓存过期自动重载（TTL 60s）
    """

    _instance: Optional["ConfigCache"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._cache: Dict[str, dict] = {}
        self._cache_ts: Dict[str, float] = {}
        self._ttl = 60  # 缓存过期时间(秒)
        self._rw_lock = threading.RLock()
        self._initialized = True
        logger.info("ConfigCache initialized, TTL=%ds", self._ttl)

    def get_agent_config(self, agent_code: str) -> Optional[dict]:
        """获取智能体配置，优先缓存"""
        with self._rw_lock:
            ts = self._cache_ts.get(agent_code, 0)
            if time.time() - ts < self._ttl and agent_code in self._cache:
                return self._cache[agent_code]

        # 缓存未命中或过期，从DB加载
        config = self._load_from_db(agent_code)
        if config:
            with self._rw_lock:
                self._cache[agent_code] = config
                self._cache_ts[agent_code] = time.time()
        return config

    def invalidate(self, agent_code: str):
        """使指定Agent配置缓存失效"""
        with self._rw_lock:
            self._cache.pop(agent_code, None)
            self._cache_ts.pop(agent_code, None)
        logger.info("ConfigCache invalidated for agent_code=%s", agent_code)

    def invalidate_all(self):
        """清空全部缓存"""
        with self._rw_lock:
            self._cache.clear()
            self._cache_ts.clear()
        logger.info("ConfigCache fully invalidated")

    def preload_all(self):
        """预加载所有启用的Agent配置"""
        try:
            with get_session() as session:
                stmt = select(LarkAgentInfo).where(LarkAgentInfo.status == 1)
                agents = session.execute(stmt).scalars().all()
                with self._rw_lock:
                    now = time.time()
                    for agent in agents:
                        cfg = self._agent_to_dict(agent)
                        self._cache[agent.agent_code] = cfg
                        self._cache_ts[agent.agent_code] = now
                logger.info("ConfigCache preloaded %d agents", len(agents))
        except Exception as e:
            logger.error("ConfigCache preload failed: %s", e)

    def _load_from_db(self, agent_code: str) -> Optional[dict]:
        """从数据库加载配置"""
        try:
            with get_session() as session:
                stmt = select(LarkAgentInfo).where(
                    LarkAgentInfo.agent_code == agent_code,
                    LarkAgentInfo.status == 1
                )
                agent = session.execute(stmt).scalar_one_or_none()
                if agent:
                    return self._agent_to_dict(agent)
                return None
        except Exception as e:
            logger.error("ConfigCache DB load failed for %s: %s", agent_code, e)
            return None

    @staticmethod
    def _agent_to_dict(agent: LarkAgentInfo) -> dict:
        """ORM对象转dict"""
        return {
            "id": agent.id,
            "agent_code": agent.agent_code,
            "agent_name": agent.agent_name,
            "industry": agent.industry,
            "language": agent.language,
            "system_prompt": agent.system_prompt,
            "industry_rules": agent.industry_rules,
            "welcome_message": agent.welcome_message,
            "model_config": agent.model_config or {},
            "knowledge_base_ids": agent.knowledge_base_ids or [],
            "quota_config": agent.quota_config or {},
            "nlu_config": agent.nlu_config or {},
            "status": agent.status,
        }


# 全局单例
config_cache = ConfigCache()
