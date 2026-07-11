"""
配额 & 权限校验中心
- 读取客户套餐信息
- 每日次数配额拦截
- 单Agent额外配额限制
- 超限返回标准提示
"""
import logging
import time
from typing import Optional, Dict
from datetime import datetime, date

from sqlalchemy import select, func

from agents.models.agent_models import LarkAgentChat, LarkAgentTokenStat
from agents.models.schemas import CustomerContext, QuotaCheckResult
from storage.database.db import get_session

logger = logging.getLogger(__name__)

# 各套餐默认每日配额
PLAN_DAILY_LIMITS: Dict[str, int] = {
    "free": 20,
    "basic": 100,
    "pro": 500,
    "flagship": 2000,
}

# 限流: 单客户每分钟最大提问次数
RATE_LIMIT_PER_MINUTE = 30


class QuotaValidator:
    """
    配额校验服务
    """

    def __init__(self):
        # 简化的内存限流计数器 (生产环境应使用Redis)
        self._rate_counters: Dict[str, list] = {}

    async def check_quota(
        self,
        customer_ctx: CustomerContext,
        agent_code: str,
        agent_config: dict,
    ) -> QuotaCheckResult:
        """
        综合配额校验
        1. 检查套餐每日配额
        2. 检查单Agent额外配额
        3. 检查分钟级限流
        """
        customer_id = customer_ctx.customer_id
        plan = customer_ctx.plan
        today = date.today()

        # 1. 套餐每日配额
        daily_limit = PLAN_DAILY_LIMITS.get(plan, PLAN_DAILY_LIMITS["free"])
        # 如果Agent有自定义配额，取较小值
        agent_quota = agent_config.get("quota_config", {})
        if agent_quota.get("daily_limit"):
            daily_limit = min(daily_limit, agent_quota["daily_limit"])

        today_usage = await self._get_today_usage(customer_id, today)
        if today_usage >= daily_limit:
            logger.warning("Quota exceeded: customer=%s plan=%s usage=%d limit=%d",
                           customer_id, plan, today_usage, daily_limit)
            return QuotaCheckResult(
                allowed=False,
                reason=f"今日对话次数已达上限({daily_limit}次)，请升级套餐或明日再试。",
                remaining=0,
                daily_limit=daily_limit,
            )

        # 2. 单Agent额外配额
        per_agent_limit = agent_quota.get("per_agent_limit")
        if per_agent_limit:
            agent_usage = await self._get_today_agent_usage(customer_id, agent_code, today)
            if agent_usage >= per_agent_limit:
                return QuotaCheckResult(
                    allowed=False,
                    reason=f"该智能体今日对话次数已达上限({per_agent_limit}次)。",
                    remaining=0,
                    daily_limit=per_agent_limit,
                )

        # 3. 分钟级限流
        if not self._check_rate_limit(customer_id):
            return QuotaCheckResult(
                allowed=False,
                reason=f"请求过于频繁，每分钟最多{RATE_LIMIT_PER_MINUTE}次，请稍后再试。",
                remaining=0,
                daily_limit=daily_limit,
            )

        remaining = daily_limit - today_usage
        return QuotaCheckResult(
            allowed=True,
            remaining=remaining,
            daily_limit=daily_limit,
        )

    async def _get_today_usage(self, customer_id: str, today: date) -> int:
        """获取今日总使用次数"""
        try:
            with get_session() as session:
                stmt = (
                    select(func.count(LarkAgentChat.id))
                    .where(
                        LarkAgentChat.customer_id == customer_id,
                        LarkAgentChat.role == "user",
                        func.date(LarkAgentChat.created_at) == today,
                    )
                )
                result = session.execute(stmt).scalar()
                return result or 0
        except Exception as e:
            logger.error("Failed to get today usage for %s: %s", customer_id, e)
            return 0  # 查询失败时临时放行

    async def _get_today_agent_usage(self, customer_id: str, agent_code: str, today: date) -> int:
        """获取今日单Agent使用次数"""
        try:
            with get_session() as session:
                stmt = (
                    select(func.count(LarkAgentChat.id))
                    .where(
                        LarkAgentChat.customer_id == customer_id,
                        LarkAgentChat.agent_code == agent_code,
                        LarkAgentChat.role == "user",
                        func.date(LarkAgentChat.created_at) == today,
                    )
                )
                result = session.execute(stmt).scalar()
                return result or 0
        except Exception as e:
            logger.error("Failed to get agent usage: %s", e)
            return 0

    def _check_rate_limit(self, customer_id: str) -> bool:
        """分钟级限流检查"""
        now = time.time()
        window_start = now - 60

        if customer_id not in self._rate_counters:
            self._rate_counters[customer_id] = []

        # 清理过期记录
        self._rate_counters[customer_id] = [
            t for t in self._rate_counters[customer_id] if t > window_start
        ]

        if len(self._rate_counters[customer_id]) >= RATE_LIMIT_PER_MINUTE:
            return False

        self._rate_counters[customer_id].append(now)
        return True

    def increment_usage(self, customer_id: str):
        """记录一次使用(用于限流计数)"""
        now = time.time()
        if customer_id not in self._rate_counters:
            self._rate_counters[customer_id] = []
        self._rate_counters[customer_id].append(now)


# 全局单例
quota_validator = QuotaValidator()
