"""Agent-Engine Pydantic 请求/响应模型"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============ 公网接口模型 ============

class ChatStreamRequest(BaseModel):
    """流式对话请求"""
    customer_id: str = Field(..., description="客户ID")
    agent_code: str = Field(..., description="智能体编码")
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="用户消息内容")
    message_type: str = Field(default="text", description="消息类型: text/image/file")
    source: str = Field(default="geo", description="来源: geo/wechat/website")
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, description="附件列表")


class ChatStreamResponse(BaseModel):
    """流式对话响应片段"""
    chat_id: str = Field(..., description="对话ID")
    content: str = Field(default="", description="响应内容片段")
    finish: bool = Field(default=False, description="是否结束")
    intent: Optional[str] = Field(default=None, description="NLU识别意图")
    model_used: Optional[str] = Field(default=None, description="使用的模型")
    error: Optional[str] = Field(default=None, description="错误信息")


class LeadSaveRequest(BaseModel):
    """留资线索提交请求"""
    customer_id: str = Field(..., description="客户ID")
    agent_code: str = Field(..., description="智能体编码")
    name: str = Field(..., description="姓名")
    phone: str = Field(..., description="手机号")
    company: Optional[str] = Field(default=None, description="公司名称")
    industry: Optional[str] = Field(default=None, description="行业")
    demand: Optional[str] = Field(default=None, description="需求描述")
    source: str = Field(default="agent_ai", description="来源标记")
    session_id: Optional[str] = Field(default=None, description="关联会话ID")


class LeadSaveResponse(BaseModel):
    """留资线索提交响应"""
    success: bool
    lead_id: Optional[str] = None
    message: str = ""


class ChatHistoryRequest(BaseModel):
    """历史对话查询请求"""
    customer_id: str
    agent_code: Optional[str] = None
    session_id: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class ChatHistoryItem(BaseModel):
    """历史对话条目"""
    chat_id: str
    role: str
    content: str
    message_type: str
    intent: Optional[str] = None
    model_used: Optional[str] = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """历史对话响应"""
    total: int
    items: List[ChatHistoryItem]
    page: int
    page_size: int


# ============ 内网管理接口模型 ============

class AgentSaveRequest(BaseModel):
    """保存/更新智能体配置请求"""
    agent_code: str = Field(..., description="智能体编码")
    agent_name: str = Field(..., description="智能体名称")
    industry: str = Field(..., description="行业")
    language: str = Field(default="zh", description="语言")
    system_prompt: str = Field(default="", description="系统Prompt")
    industry_rules: str = Field(default="", description="行业规则")
    welcome_message: str = Field(default="", description="欢迎语")
    model_route_config: Dict[str, Any] = Field(default_factory=dict, description="模型路由配置")
    knowledge_base_ids: List[str] = Field(default_factory=list, description="知识库ID列表")
    quota_config: Dict[str, Any] = Field(default_factory=dict, description="配额配置")
    nlu_config: Dict[str, Any] = Field(default_factory=dict, description="NLU配置")
    status: int = Field(default=1, description="状态")


class AgentInfoResponse(BaseModel):
    """智能体信息响应"""
    id: int
    agent_code: str
    agent_name: str
    industry: str
    language: str
    system_prompt: str
    industry_rules: str
    welcome_message: str
    model_route_config: dict
    knowledge_base_ids: list
    quota_config: dict
    nlu_config: dict
    status: int
    created_at: datetime
    updated_at: datetime


class AgentListResponse(BaseModel):
    """智能体列表响应"""
    total: int
    items: List[AgentInfoResponse]


class ChatListRequest(BaseModel):
    """对话记录查询请求"""
    customer_id: Optional[str] = None
    agent_code: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class TokenStatRequest(BaseModel):
    """Token统计查询请求"""
    customer_id: Optional[str] = None
    agent_code: Optional[str] = None
    model_name: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class TokenStatItem(BaseModel):
    """Token统计条目"""
    customer_id: str
    agent_code: str
    model_name: str
    stat_date: date
    total_input_tokens: int
    total_output_tokens: int
    total_requests: int


class TokenStatResponse(BaseModel):
    """Token统计响应"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    items: List[TokenStatItem] = []


class ModelRouteSetRequest(BaseModel):
    """模型路由配置请求"""
    agent_code: str = Field(..., description="智能体编码")
    primary_model: str = Field(..., description="主模型")
    fallback_models: List[str] = Field(default_factory=list, description="备用模型列表")
    timeout: int = Field(default=30, description="超时时间(秒)")


class QuotaCheckResult(BaseModel):
    """配额校验结果"""
    allowed: bool
    reason: Optional[str] = None
    remaining: Optional[int] = None
    daily_limit: Optional[int] = None


# ============ 内部服务模型 ============

class CustomerContext(BaseModel):
    """客户上下文信息(从Token解析)"""
    customer_id: str
    plan: str = Field(default="free", description="套餐: free/basic/pro/flagship")
    industry: str = Field(default="", description="行业标签")
    permissions: List[str] = Field(default_factory=list, description="权限列表")


class NLUIntent(BaseModel):
    """NLU意图识别结果"""
    intent: str = Field(default="qa", description="意图类型: qa/lead_guide/wechat_invoke/human_transfer")
    confidence: float = Field(default=0.0, description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")
