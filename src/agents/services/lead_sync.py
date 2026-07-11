"""
线索同步模块
- 接收前端留资数据
- 调用拉克力中台接口写入线索池
- 线索来源标记: AgentAI对话留资
- 自动关联客户ID、行业、对话上下文
"""
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy import insert

from agents.models.schemas import LeadSaveRequest, LeadSaveResponse

logger = logging.getLogger(__name__)

# 拉克力线索写入接口地址（内网）
LARK_LEADS_API_URL = "/api/lark/leads/create"


class LeadSyncService:
    """
    线索同步服务
    负责将AI对话中的留资数据同步到拉克力线索池
    """

    def __init__(self):
        self._lark_api_base = ""  # 内网拉克力后台地址，从环境变量获取
        logger.info("LeadSyncService initialized")

    async def save_lead(self, request: LeadSaveRequest, agent_config: Optional[dict] = None) -> LeadSaveResponse:
        """
        保存留资线索
        1. 构建线索数据
        2. 调用拉克力中台接口写入
        3. 返回结果
        """
        lead_id = f"lead_{uuid.uuid4().hex[:16]}"
        industry = request.industry or (agent_config.get("industry", "") if agent_config else "")

        lead_data = {
            "lead_id": lead_id,
            "customer_id": request.customer_id,
            "name": request.name,
            "phone": request.phone,
            "company": request.company or "",
            "industry": industry,
            "demand": request.demand or "",
            "source": "AgentAI对话留资",
            "agent_code": request.agent_code,
            "session_id": request.session_id or "",
            "created_at": datetime.utcnow().isoformat(),
        }

        # 尝试调用拉克力中台接口
        success = await self._push_to_lark(lead_data)

        if success:
            logger.info("Lead saved: lead_id=%s customer=%s", lead_id, request.customer_id)
            return LeadSaveResponse(success=True, lead_id=lead_id, message="线索提交成功")
        else:
            logger.warning("Lead push to Lark failed, lead_id=%s", lead_id)
            return LeadSaveResponse(success=False, message="线索提交失败，请稍后重试")

    async def _push_to_lark(self, lead_data: dict) -> bool:
        """
        推送线索到拉克力中台
        调用拉克力后台线索写入接口
        """
        import httpx

        api_base = self._lark_api_base or "http://lark-admin.internal"
        url = f"{api_base}{LARK_LEADS_API_URL}"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=lead_data)
                if resp.status_code == 200:
                    result = resp.json()
                    return result.get("success", False)
                else:
                    logger.error("Lark leads API returned %d: %s", resp.status_code, resp.text)
                    return False
        except Exception as e:
            logger.error("Failed to push lead to Lark: %s", e)
            # 拉克力接口不可用时，记录本地日志作为兜底
            logger.info("Lead data (fallback log): %s", lead_data)
            return False


# 全局单例
lead_sync = LeadSyncService()
