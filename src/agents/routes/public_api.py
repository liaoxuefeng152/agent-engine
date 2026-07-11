"""
公网 API 路由
1. WebSocket /api/agent/chat/stream - 流式对话主接口
2. POST /api/agent/lead/save - 留资线索提交
3. GET /api/agent/chat/history - 获取客户历史对话
4. GET /api/agent/knowledge/file - 行业资料下载
"""
import json
import logging
import uuid
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, AIMessage

from agents.models.schemas import (
    ChatStreamRequest, LeadSaveRequest, CustomerContext,
)
from agents.services.config_cache import config_cache
from agents.services.model_router import model_router
from agents.services.rag_engine import rag_engine
from agents.services.prompt_manager import prompt_manager
from agents.services.quota_validator import quota_validator
from agents.services.nlu_engine import nlu_engine
from agents.services.lead_sync import lead_sync
from agents.services.token_counter import token_counter, chat_persistence
from agents.middleware.auth import parse_websocket_token, filter_sensitive_words
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["public"])


# ============ WebSocket 流式对话 ============

@router.websocket("/chat/stream")
async def websocket_chat_stream(websocket: WebSocket):
    """
    WebSocket 流式对话主接口
    - 支持文字提问、图片识别、附件上下文理解
    - 逐字流式输出
    - 会话上下文记忆
    - 异常断连自动重连、心跳保活
    """
    # 1. 鉴权
    customer_ctx = parse_websocket_token(websocket)
    if customer_ctx is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("WebSocket connected: customer=%s", customer_ctx.customer_id)

    try:
        while True:
            # 2. 接收消息
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
                continue

            message = data.get("message", "").strip()
            agent_code = data.get("agent_code", "")
            session_id = data.get("session_id", f"sess_{uuid.uuid4().hex[:12]}")
            source = data.get("source", "geo")
            message_type = data.get("message_type", "text")

            if not message or not agent_code:
                await websocket.send_json({"error": "message and agent_code are required"})
                continue

            # 敏感词过滤
            message = filter_sensitive_words(message)

            # 3. 加载Agent配置
            agent_config = config_cache.get_agent_config(agent_code)
            if not agent_config:
                await websocket.send_json({"error": f"Agent '{agent_code}' not found or disabled"})
                continue

            # 4. 配额校验
            quota_result = await quota_validator.check_quota(customer_ctx, agent_code, agent_config)
            if not quota_result.allowed:
                await websocket.send_json({
                    "chat_id": "",
                    "content": quota_result.reason or "配额不足",
                    "finish": True,
                    "error": "quota_exceeded",
                })
                continue

            # 5. NLU 意图识别
            intent = nlu_engine.recognize(message, agent_config.get("nlu_config"))

            # 如果命中特殊意图，直接返回预设回复
            if intent.intent != "qa":
                intent_response = nlu_engine.get_intent_response(intent)
                chat_id = await chat_persistence.save_message(
                    customer_id=customer_ctx.customer_id,
                    agent_code=agent_code,
                    session_id=session_id,
                    role="user",
                    content=message,
                    message_type=message_type,
                    intent=intent.intent,
                    source=source,
                )
                if intent_response:
                    await chat_persistence.save_message(
                        customer_id=customer_ctx.customer_id,
                        agent_code=agent_code,
                        session_id=session_id,
                        role="assistant",
                        content=intent_response,
                        intent=intent.intent,
                        source=source,
                    )
                    await websocket.send_json({
                        "chat_id": chat_id,
                        "content": intent_response,
                        "finish": True,
                        "intent": intent.intent,
                        "command": nlu_engine.get_intent_command(intent),
                    })

                # 如果是留资引导，额外返回引导信息
                if intent.intent == "lead_guide":
                    await websocket.send_json({
                        "chat_id": "",
                        "content": "",
                        "finish": True,
                        "intent": "lead_guide_form",
                        "show_lead_form": True,
                    })
                continue

            # 6. RAG 知识库检索
            kb_ids = agent_config.get("knowledge_base_ids", [])
            rag_context = ""
            if kb_ids:
                results = rag_engine.search(message, kb_ids, top_k=5, threshold=0.5)
                rag_context = rag_engine.format_context(results)

            # 7. 获取历史上下文
            recent = await chat_persistence.get_recent_context(session_id, limit=10)
            history_messages = []
            for msg in recent:
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))

            # 8. 构建Prompt
            messages = prompt_manager.build_messages(
                agent_code=agent_code,
                user_message=message,
                history=history_messages,
                rag_context=rag_context,
            )

            # 9. 保存用户消息
            ctx = new_context(method="chat_stream")
            user_chat_id = await chat_persistence.save_message(
                customer_id=customer_ctx.customer_id,
                agent_code=agent_code,
                session_id=session_id,
                role="user",
                content=message,
                message_type=message_type,
                intent="qa",
                source=source,
            )

            # 10. 流式调用模型
            full_response = ""
            model_used = ""
            total_input_tokens = 0
            total_output_tokens = 0

            async for chunk_content, is_final, model, usage in model_router.call_stream(
                messages=messages,
                model_config=agent_config.get("model_config", {}),
                ctx=ctx,
            ):
                if chunk_content:
                    full_response += chunk_content
                    await websocket.send_json({
                        "chat_id": user_chat_id,
                        "content": chunk_content,
                        "finish": False,
                        "model_used": model,
                    })

                if is_final:
                    model_used = model
                    if usage:
                        total_input_tokens = usage.get("prompt_tokens", 0)
                        total_output_tokens = usage.get("completion_tokens", 0)

            # 11. 保存AI回复
            await chat_persistence.save_message(
                customer_id=customer_ctx.customer_id,
                agent_code=agent_code,
                session_id=session_id,
                role="assistant",
                content=full_response,
                intent="qa",
                model_used=model_used,
                token_input=total_input_tokens,
                token_output=total_output_tokens,
                source=source,
            )

            # 12. Token统计
            if model_used:
                await token_counter.record_usage(
                    customer_id=customer_ctx.customer_id,
                    agent_code=agent_code,
                    model_name=model_used,
                    token_input=total_input_tokens,
                    token_output=total_output_tokens,
                )

            # 13. 发送结束标记
            await websocket.send_json({
                "chat_id": user_chat_id,
                "content": "",
                "finish": True,
                "model_used": model_used,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: customer=%s", customer_ctx.customer_id)
    except asyncio.TimeoutError:
        logger.info("WebSocket timeout: customer=%s", customer_ctx.customer_id)
        try:
            await websocket.close(code=1000, reason="Session timeout")
        except Exception:
            pass
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"error": str(e), "finish": True})
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass


# ============ 留资线索提交 ============

@router.post("/lead/save")
async def save_lead(request: Request):
    """留资线索提交"""
    customer_ctx = getattr(request.state, "customer_ctx", None)
    if not customer_ctx:
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    lead_req = LeadSaveRequest(**body)
    lead_req.customer_id = customer_ctx.customer_id

    agent_config = config_cache.get_agent_config(lead_req.agent_code)
    result = await lead_sync.save_lead(lead_req, agent_config)

    return JSONResponse(content=result.model_dump())


# ============ 历史对话查询 ============

@router.get("/chat/history")
async def get_chat_history(
    request: Request,
    agent_code: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """获取客户历史对话"""
    customer_ctx = getattr(request.state, "customer_ctx", None)
    if not customer_ctx:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = await chat_persistence.get_history(
        customer_id=customer_ctx.customer_id,
        agent_code=agent_code,
        session_id=session_id,
        page=page,
        page_size=page_size,
    )
    return JSONResponse(content=result)


# ============ 行业资料下载 ============

@router.get("/knowledge/file")
async def download_knowledge_file(
    request: Request,
    kb_id: str = Query(..., description="知识库ID"),
    filename: str = Query(..., description="文件名"),
):
    """行业资料下载"""
    customer_ctx = getattr(request.state, "customer_ctx", None)
    if not customer_ctx:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 从对象存储获取文件
    # 生产环境应通过S3/OSS获取
    import os
    file_path = os.path.join("/tmp", "knowledge", kb_id, filename)
    if os.path.exists(file_path):
        from fastapi.responses import FileResponse
        return FileResponse(file_path, filename=filename)

    raise HTTPException(status_code=404, detail="File not found")
