"""
动态 Prompt 管理服务
- 读取 Agent 配置，拼接人设Prompt + 行业规则 + 上下文 + 知识库片段
- 支持中英文双语言自动适配
- 配置热更新实时生效
"""
import logging
from typing import Optional, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from agents.services.config_cache import config_cache
from agents.services.rag_engine import rag_engine

logger = logging.getLogger(__name__)

# 中英文默认Prompt模板
DEFAULT_SYSTEM_PROMPT_ZH = (
    "你是拉克力AI智能助手，一个专业、友好的AI助手。"
    "你的职责是为用户提供准确、有用的信息和服务。"
    "请用中文回答用户的问题，保持专业和礼貌。"
)

DEFAULT_SYSTEM_PROMPT_EN = (
    "You are LarkAI Assistant, a professional and friendly AI assistant. "
    "Your role is to provide users with accurate and helpful information and services. "
    "Please answer user questions in English, maintaining professionalism and courtesy."
)


class PromptManager:
    """
    动态 Prompt 管理器
    负责拼接最终的 System Prompt：
    [人设Prompt] + [行业规则] + [知识库上下文] + [对话约束]
    """

    def build_system_prompt(
        self,
        agent_code: str,
        rag_context: str = "",
        extra_context: str = "",
    ) -> str:
        """
        构建完整的系统Prompt
        """
        config = config_cache.get_agent_config(agent_code)
        if not config:
            logger.warning("Agent config not found for %s, using default", agent_code)
            return DEFAULT_SYSTEM_PROMPT_ZH

        language = config.get("language", "zh")
        custom_prompt = config.get("system_prompt", "")
        industry_rules = config.get("industry_rules", "")

        # 1. 基础人设
        if custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = DEFAULT_SYSTEM_PROMPT_EN if language == "en" else DEFAULT_SYSTEM_PROMPT_ZH

        # 2. 行业规则
        rules_section = ""
        if industry_rules:
            rules_section = f"\n\n## 行业规则\n{industry_rules}"

        # 3. 知识库上下文
        rag_section = ""
        if rag_context:
            rag_section = f"\n\n## 参考资料（请优先基于以下资料回答）\n{rag_context}"

        # 4. 额外上下文(如会话摘要等)
        extra_section = ""
        if extra_context:
            extra_section = f"\n\n## 补充信息\n{extra_context}"

        # 5. 通用约束
        if language == "en":
            constraint = (
                "\n\n## Constraints\n"
                "- Answer based on the provided knowledge base first.\n"
                "- If you don't know the answer, say so honestly.\n"
                "- Keep responses concise and professional.\n"
                "- Do not fabricate information."
            )
        else:
            constraint = (
                "\n\n## 约束\n"
                "- 优先基于提供的知识库资料回答。\n"
                "- 如果不确定答案，请如实告知。\n"
                "- 回答简洁专业，不编造信息。\n"
                "- 如涉及价格、合作等商务问题，引导用户留下联系方式。"
            )

        final_prompt = base_prompt + rules_section + rag_section + extra_section + constraint
        return final_prompt

    def build_messages(
        self,
        agent_code: str,
        user_message: str,
        history: Optional[List[BaseMessage]] = None,
        rag_context: str = "",
    ) -> List[BaseMessage]:
        """
        构建完整的消息列表
        """
        system_prompt = self.build_system_prompt(agent_code, rag_context=rag_context)
        messages = [SystemMessage(content=system_prompt)]

        # 添加历史对话
        if history:
            messages.extend(history[-10:])  # 最多保留最近10轮

        # 添加当前用户消息
        messages.append(HumanMessage(content=user_message))

        return messages


# 全局单例
prompt_manager = PromptManager()
