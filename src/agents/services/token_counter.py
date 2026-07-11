"""
Token 统计与数据持久化模块
- 每轮对话精准统计输入/输出Token
- 按客户、Agent、模型多维度统计入库
- 超限预警
- 对话记录持久化
"""
import logging
import uuid
from datetime import datetime, date
from typing import Optional

from sqlalchemy import select, func, insert, update

from agents.models.agent_models import LarkAgentChat, LarkAgentTokenStat
from storage.database.db import get_session

logger = logging.getLogger(__name__)


class TokenCounter:
    """Token 统计服务"""

    async def record_usage(
        self,
        customer_id: str,
        agent_code: str,
        model_name: str,
        token_input: int,
        token_output: int,
    ):
        """
        记录Token消耗（按天聚合upsert）
        """
        today = date.today()
        try:
            with get_session() as session:
                # 查找今日记录
                stmt = select(LarkAgentTokenStat).where(
                    LarkAgentTokenStat.customer_id == customer_id,
                    LarkAgentTokenStat.agent_code == agent_code,
                    LarkAgentTokenStat.model_name == model_name,
                    LarkAgentTokenStat.stat_date == today,
                )
                existing = session.execute(stmt).scalar_one_or_none()

                if existing:
                    existing.total_input_tokens += token_input
                    existing.total_output_tokens += token_output
                    existing.total_requests += 1
                else:
                    new_stat = LarkAgentTokenStat(
                        customer_id=customer_id,
                        agent_code=agent_code,
                        model_name=model_name,
                        stat_date=today,
                        total_input_tokens=token_input,
                        total_output_tokens=token_output,
                        total_requests=1,
                    )
                    session.add(new_stat)

                session.commit()
                logger.debug("Token usage recorded: customer=%s agent=%s model=%s in=%d out=%d",
                             customer_id, agent_code, model_name, token_input, token_output)
        except Exception as e:
            logger.error("Failed to record token usage: %s", e)

    async def get_stats(
        self,
        customer_id: Optional[str] = None,
        agent_code: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict:
        """查询Token统计"""
        try:
            with get_session() as session:
                stmt = select(LarkAgentTokenStat)

                if customer_id:
                    stmt = stmt.where(LarkAgentTokenStat.customer_id == customer_id)
                if agent_code:
                    stmt = stmt.where(LarkAgentTokenStat.agent_code == agent_code)
                if model_name:
                    stmt = stmt.where(LarkAgentTokenStat.model_name == model_name)
                if start_date:
                    stmt = stmt.where(LarkAgentTokenStat.stat_date >= start_date)
                if end_date:
                    stmt = stmt.where(LarkAgentTokenStat.stat_date <= end_date)

                stmt = stmt.order_by(LarkAgentTokenStat.stat_date.desc())
                results = session.execute(stmt).scalars().all()

                total_in = sum(r.total_input_tokens for r in results)
                total_out = sum(r.total_output_tokens for r in results)
                total_req = sum(r.total_requests for r in results)

                items = [
                    {
                        "customer_id": r.customer_id,
                        "agent_code": r.agent_code,
                        "model_name": r.model_name,
                        "stat_date": r.stat_date.isoformat() if r.stat_date else "",
                        "total_input_tokens": r.total_input_tokens,
                        "total_output_tokens": r.total_output_tokens,
                        "total_requests": r.total_requests,
                    }
                    for r in results
                ]

                return {
                    "total_input_tokens": total_in,
                    "total_output_tokens": total_out,
                    "total_requests": total_req,
                    "items": items,
                }
        except Exception as e:
            logger.error("Failed to get token stats: %s", e)
            return {"total_input_tokens": 0, "total_output_tokens": 0, "total_requests": 0, "items": []}


class ChatPersistence:
    """对话数据持久化服务"""

    async def save_message(
        self,
        customer_id: str,
        agent_code: str,
        session_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        intent: Optional[str] = None,
        model_used: Optional[str] = None,
        token_input: int = 0,
        token_output: int = 0,
        source: str = "geo",
    ) -> str:
        """保存对话消息，返回chat_id"""
        chat_id = f"chat_{uuid.uuid4().hex[:16]}"
        try:
            with get_session() as session:
                chat = LarkAgentChat(
                    chat_id=chat_id,
                    customer_id=customer_id,
                    agent_code=agent_code,
                    session_id=session_id,
                    role=role,
                    content=content,
                    message_type=message_type,
                    intent=intent,
                    model_used=model_used,
                    token_input=token_input,
                    token_output=token_output,
                    source=source,
                )
                session.add(chat)
                session.commit()
                logger.debug("Chat saved: chat_id=%s role=%s", chat_id, role)
                return chat_id
        except Exception as e:
            logger.error("Failed to save chat message: %s", e)
            return ""

    async def get_history(
        self,
        customer_id: str,
        agent_code: Optional[str] = None,
        session_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """查询对话历史"""
        try:
            with get_session() as session:
                stmt = select(LarkAgentChat).where(LarkAgentChat.customer_id == customer_id)

                if agent_code:
                    stmt = stmt.where(LarkAgentChat.agent_code == agent_code)
                if session_id:
                    stmt = stmt.where(LarkAgentChat.session_id == session_id)

                # 总数
                count_stmt = select(func.count()).select_from(stmt.subquery())
                total = session.execute(count_stmt).scalar() or 0

                # 分页
                stmt = stmt.order_by(LarkAgentChat.created_at.desc())
                stmt = stmt.offset((page - 1) * page_size).limit(page_size)
                results = session.execute(stmt).scalars().all()

                items = [
                    {
                        "chat_id": r.chat_id,
                        "role": r.role,
                        "content": r.content,
                        "message_type": r.message_type,
                        "intent": r.intent,
                        "model_used": r.model_used,
                        "created_at": r.created_at.isoformat() if r.created_at else "",
                    }
                    for r in results
                ]

                return {"total": total, "items": items, "page": page, "page_size": page_size}
        except Exception as e:
            logger.error("Failed to get chat history: %s", e)
            return {"total": 0, "items": [], "page": page, "page_size": page_size}

    async def get_recent_context(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list:
        """获取最近的对话上下文(用于拼接历史)"""
        try:
            with get_session() as session:
                stmt = (
                    select(LarkAgentChat)
                    .where(LarkAgentChat.session_id == session_id)
                    .order_by(LarkAgentChat.created_at.desc())
                    .limit(limit)
                )
                results = session.execute(stmt).scalars().all()
                # 反转为时间正序
                results = list(reversed(results))
                return [
                    {"role": r.role, "content": r.content}
                    for r in results
                ]
        except Exception as e:
            logger.error("Failed to get recent context: %s", e)
            return []


# 全局单例
token_counter = TokenCounter()
chat_persistence = ChatPersistence()
