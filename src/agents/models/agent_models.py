"""
Agent-Engine 数据库 ORM 模型
三张核心表: lark_agent_info / lark_agent_chat / lark_agent_token_stat
"""
import datetime
from sqlalchemy import (
    BigInteger, DateTime, Integer, String, Text, JSON,
    Index, text, func
)
from sqlalchemy.orm import Mapped, mapped_column
from storage.database.shared.model import Base


class LarkAgentInfo(Base):
    """智能体配置表"""
    __tablename__ = "lark_agent_info"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="智能体唯一编码")
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="智能体名称")
    industry: Mapped[str] = mapped_column(String(64), nullable=False, comment="所属行业标签")
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="zh", comment="语言: zh/en")

    # Prompt 配置
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="人设系统Prompt")
    industry_rules: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="行业规则Prompt")
    welcome_message: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="欢迎语")

    # 模型路由配置 (JSON)
    # 格式: {"primary": "doubao-seed-2-0-lite-260215", "fallback": ["doubao-seed-2-0-mini-260215"], "timeout": 30}
    model_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, comment="模型路由配置")

    # 知识库绑定
    knowledge_base_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list, comment="绑定的知识库ID列表")

    # 配额配置
    quota_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, comment="配额配置: daily_limit, per_agent_limit")

    # NLU 意图关键词配置 (JSON)
    nlu_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, comment="NLU意图关键词配置")

    # 状态
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="状态: 0=禁用 1=启用")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )

    __table_args__ = (
        Index("idx_agent_industry", "industry"),
        Index("idx_agent_status", "status"),
    )


class LarkAgentChat(Base):
    """对话记录表"""
    __tablename__ = "lark_agent_chat"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="对话唯一ID")
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="客户ID")
    agent_code: Mapped[str] = mapped_column(String(64), nullable=False, comment="智能体编码")
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="会话ID(上下文关联)")

    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="角色: user/assistant/system")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息内容")

    # 消息元数据
    message_type: Mapped[str] = mapped_column(String(16), nullable=False, default="text", comment="消息类型: text/image/file")
    intent: Mapped[str] = mapped_column(String(32), nullable=True, comment="NLU识别意图: qa/lead/wechat_transfer/human_transfer")
    model_used: Mapped[str] = mapped_column(String(64), nullable=True, comment="实际使用的模型")
    token_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="输入Token数")
    token_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="输出Token数")

    # 来源
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="geo", comment="来源: geo/wechat/website")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), comment="创建时间"
    )

    __table_args__ = (
        Index("idx_chat_customer", "customer_id"),
        Index("idx_chat_agent", "agent_code"),
        Index("idx_chat_session", "session_id"),
        Index("idx_chat_created", "created_at"),
    )


class LarkAgentTokenStat(Base):
    """Token消耗统计表"""
    __tablename__ = "lark_agent_token_stat"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="客户ID")
    agent_code: Mapped[str] = mapped_column(String(64), nullable=False, comment="智能体编码")
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, comment="模型名称")

    stat_date: Mapped[datetime.date] = mapped_column(
        DateTime, nullable=False, comment="统计日期(按天聚合)"
    )
    total_input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment="总输入Token")
    total_output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment="总输出Token")
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="总请求次数")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )

    __table_args__ = (
        Index("idx_stat_customer_date", "customer_id", "stat_date"),
        Index("idx_stat_agent_date", "agent_code", "stat_date"),
        Index("idx_stat_model_date", "model_name", "stat_date"),
    )
