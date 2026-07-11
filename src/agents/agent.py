"""
Agent-Engine 核心 Agent 入口
提供 build_agent 方法，构建基于 LangChain 的 AI 对话 Agent
集成所有核心服务：模型路由、RAG、Prompt管理、配额校验、NLU等
"""
import os
import json
import logging
from typing import Annotated, Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from coze_coding_utils.runtime_ctx.context import default_headers

from storage.memory.memory_saver import get_memory_saver

logger = logging.getLogger(__name__)

LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40


def _windowed_messages(old, new):
    """滑动窗口: 只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]  # type: ignore


class AgentState(MessagesState):
    messages: Annotated[list[AnyMessage], _windowed_messages]


def build_agent(ctx=None):
    """
    构建 Agent-Engine Agent 实例
    - 使用 create_agent 创建
    - 集成短期记忆(checkpointer)
    - 配置模型路由
    """
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")

    llm = ChatOpenAI(
        model=cfg['config'].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg['config'].get('temperature', 0.7),
        streaming=True,
        timeout=cfg['config'].get('timeout', 600),
        extra_body={
            "thinking": {
                "type": cfg['config'].get('thinking', 'disabled')
            }
        },
        default_headers=default_headers(ctx) if ctx else {},
    )

    # 获取 checkpointer
    try:
        checkpointer = get_memory_saver()
    except Exception as e:
        logger.warning("Failed to get memory saver, using fallback: %s", e)
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    agent = create_agent(
        model=llm,
        system_prompt=cfg.get("sp", ""),
        tools=[],
        checkpointer=checkpointer,
        state_schema=AgentState,
    )

    return agent
