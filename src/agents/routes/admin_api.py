"""
内网管理 API 路由（仅内网可访问）
1. POST /api/agent/admin/save - 保存/更新智能体配置
2. GET /api/agent/admin/list - Agent列表查询
3. GET /api/agent/admin/chat/list - 全量对话记录查询
4. GET /api/agent/admin/stat/token - Token统计数据
5. POST /api/agent/admin/vector/reload - 向量库重载刷新
6. POST /api/agent/admin/model/set - 模型路由配置
"""
import logging
from typing import Optional
from datetime import date, datetime

from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, update

from agents.models.agent_models import LarkAgentInfo, LarkAgentChat, LarkAgentTokenStat
from agents.models.schemas import (
    AgentSaveRequest, AgentInfoResponse, AgentListResponse,
    ChatListRequest, TokenStatRequest, ModelRouteSetRequest,
)
from agents.services.config_cache import config_cache
from agents.services.rag_engine import rag_engine
from agents.services.token_counter import token_counter
from storage.database.db import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent/admin", tags=["admin"])


# ============ 保存/更新智能体配置 ============

@router.post("/save")
async def save_agent(request: Request):
    """保存或更新智能体配置"""
    body = await request.json()
    save_req = AgentSaveRequest(**body)

    try:
        with get_session() as session:
            # 查找是否已存在
            stmt = select(LarkAgentInfo).where(LarkAgentInfo.agent_code == save_req.agent_code)
            existing = session.execute(stmt).scalar_one_or_none()

            if existing:
                # 更新
                existing.agent_name = save_req.agent_name
                existing.industry = save_req.industry
                existing.language = save_req.language
                existing.system_prompt = save_req.system_prompt
                existing.industry_rules = save_req.industry_rules
                existing.welcome_message = save_req.welcome_message
                existing.model_config = save_req.model_route_config
                existing.knowledge_base_ids = save_req.knowledge_base_ids
                existing.quota_config = save_req.quota_config
                existing.nlu_config = save_req.nlu_config
                existing.status = save_req.status
            else:
                # 新建
                agent = LarkAgentInfo(
                    agent_code=save_req.agent_code,
                    agent_name=save_req.agent_name,
                    industry=save_req.industry,
                    language=save_req.language,
                    system_prompt=save_req.system_prompt,
                    industry_rules=save_req.industry_rules,
                    welcome_message=save_req.welcome_message,
                    model_config=save_req.model_route_config,
                    knowledge_base_ids=save_req.knowledge_base_ids,
                    quota_config=save_req.quota_config,
                    nlu_config=save_req.nlu_config,
                    status=save_req.status,
                )
                session.add(agent)

            session.commit()

            # 刷新缓存
            config_cache.invalidate(save_req.agent_code)

        logger.info("Agent saved: agent_code=%s", save_req.agent_code)
        return JSONResponse(content={"success": True, "message": "Agent配置保存成功"})

    except Exception as e:
        logger.error("Failed to save agent: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


# ============ Agent列表查询 ============

@router.get("/list")
async def list_agents(
    industry: Optional[str] = Query(None),
    status: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Agent列表查询"""
    try:
        with get_session() as session:
            stmt = select(LarkAgentInfo)
            if industry:
                stmt = stmt.where(LarkAgentInfo.industry == industry)
            if status is not None:
                stmt = stmt.where(LarkAgentInfo.status == status)

            # 总数
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = session.execute(count_stmt).scalar() or 0

            # 分页
            stmt = stmt.order_by(LarkAgentInfo.updated_at.desc())
            stmt = stmt.offset((page - 1) * page_size).limit(page_size)
            results = session.execute(stmt).scalars().all()

            items = []
            for r in results:
                items.append({
                    "id": r.id,
                    "agent_code": r.agent_code,
                    "agent_name": r.agent_name,
                    "industry": r.industry,
                    "language": r.language,
                    "system_prompt": r.system_prompt,
                    "industry_rules": r.industry_rules,
                    "welcome_message": r.welcome_message,
                    "model_config": r.model_config,
                    "knowledge_base_ids": r.knowledge_base_ids,
                    "quota_config": r.quota_config,
                    "nlu_config": r.nlu_config,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                })

            return JSONResponse(content={"total": total, "items": items, "page": page, "page_size": page_size})

    except Exception as e:
        logger.error("Failed to list agents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============ 全量对话记录查询 ============

@router.get("/chat/list")
async def list_chats(
    customer_id: Optional[str] = Query(None),
    agent_code: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """全量对话记录查询"""
    try:
        with get_session() as session:
            stmt = select(LarkAgentChat)
            if customer_id:
                stmt = stmt.where(LarkAgentChat.customer_id == customer_id)
            if agent_code:
                stmt = stmt.where(LarkAgentChat.agent_code == agent_code)
            if start_time:
                stmt = stmt.where(LarkAgentChat.created_at >= datetime.fromisoformat(start_time))
            if end_time:
                stmt = stmt.where(LarkAgentChat.created_at <= datetime.fromisoformat(end_time))

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = session.execute(count_stmt).scalar() or 0

            stmt = stmt.order_by(LarkAgentChat.created_at.desc())
            stmt = stmt.offset((page - 1) * page_size).limit(page_size)
            results = session.execute(stmt).scalars().all()

            items = [
                {
                    "chat_id": r.chat_id,
                    "customer_id": r.customer_id,
                    "agent_code": r.agent_code,
                    "session_id": r.session_id,
                    "role": r.role,
                    "content": r.content,
                    "message_type": r.message_type,
                    "intent": r.intent,
                    "model_used": r.model_used,
                    "token_input": r.token_input,
                    "token_output": r.token_output,
                    "source": r.source,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                }
                for r in results
            ]

            return JSONResponse(content={"total": total, "items": items, "page": page, "page_size": page_size})

    except Exception as e:
        logger.error("Failed to list chats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============ Token统计数据 ============

@router.get("/stat/token")
async def get_token_stats(
    customer_id: Optional[str] = Query(None),
    agent_code: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Token统计数据"""
    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None

    result = await token_counter.get_stats(
        customer_id=customer_id,
        agent_code=agent_code,
        model_name=model_name,
        start_date=sd,
        end_date=ed,
    )
    return JSONResponse(content=result)


# ============ 向量库重载刷新 ============

@router.post("/vector/reload")
async def reload_vector(request: Request):
    """向量库重载刷新"""
    body = await request.json()
    kb_id = body.get("kb_id", "")

    if not kb_id:
        # 重载所有
        rag_engine.reload_knowledge_base("__all__")
        return JSONResponse(content={"success": True, "message": "All vector indexes reloaded"})

    rag_engine.reload_knowledge_base(kb_id)
    return JSONResponse(content={"success": True, "message": f"Vector index for kb_id={kb_id} reloaded"})


# ============ 模型路由配置 ============

@router.post("/model/set")
async def set_model_route(request: Request):
    """设置模型路由配置"""
    body = await request.json()
    route_req = ModelRouteSetRequest(**body)

    try:
        with get_session() as session:
            stmt = select(LarkAgentInfo).where(LarkAgentInfo.agent_code == route_req.agent_code)
            agent = session.execute(stmt).scalar_one_or_none()

            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent '{route_req.agent_code}' not found")

            # 更新模型配置
            agent.model_config = {
                "primary": route_req.primary_model,
                "fallback": route_req.fallback_models,
                "timeout": route_req.timeout,
            }
            session.commit()

            # 刷新缓存
            config_cache.invalidate(route_req.agent_code)

        logger.info("Model route updated: agent=%s primary=%s", route_req.agent_code, route_req.primary_model)
        return JSONResponse(content={"success": True, "message": "模型路由配置更新成功"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to set model route: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
