"""
NLU 意图业务联动引擎
- 意向词命中 → 返回企微唤起指令
- 咨询价格/合作 → 触发留资引导话术
- 人工咨询关键词 → 自动转接人工客服
- 全部后端识别，无需前端判断
"""
import logging
import re
from typing import Optional, Dict, List

from agents.models.schemas import NLUIntent

logger = logging.getLogger(__name__)


# 默认意图关键词配置
DEFAULT_NLU_CONFIG: Dict[str, Dict] = {
    "wechat_invoke": {
        "keywords": [
            "加微信", "企业微信", "加企微", "微信客服", "加你微信",
            "微信号", "扫码", "加微", "weixin", "wechat",
        ],
        "response_template": "好的，我为您唤起企业微信客服，请稍等片刻。",
        "command": "invoke_wechat",
    },
    "lead_guide": {
        "keywords": [
            "价格", "报价", "多少钱", "费用", "合作", "代理",
            "定制", "方案", "采购", "批量", "大客户",
            "怎么买", "如何购买", "套餐", "收费标准",
        ],
        "response_template": (
            "感谢您的关注！为了给您提供更精准的方案和报价，"
            "方便留下您的联系方式吗？我们的专业顾问会尽快与您联系。"
        ),
        "command": "collect_lead",
    },
    "human_transfer": {
        "keywords": [
            "人工客服", "转人工", "真人", "真人客服", "找人工",
            "人工服务", "投诉", "转接", "找客服",
        ],
        "response_template": "好的，正在为您转接人工客服，请稍候。",
        "command": "transfer_human",
    },
}


class NLUEngine:
    """
    NLU 意图识别引擎
    基于关键词匹配 + 规则引擎实现意图识别
    """

    def __init__(self):
        self._default_config = DEFAULT_NLU_CONFIG
        logger.info("NLUEngine initialized")

    def recognize(
        self,
        user_message: str,
        agent_nlu_config: Optional[Dict] = None,
    ) -> NLUIntent:
        """
        识别用户消息意图
        优先使用Agent自定义配置，否则使用默认配置
        """
        message_lower = user_message.lower().strip()
        if not message_lower:
            return NLUIntent(intent="qa", confidence=0.0)

        # 合并配置: Agent自定义 > 默认
        nlu_config = self._merge_config(agent_nlu_config)

        # 按优先级检测意图: human_transfer > wechat_invoke > lead_guide
        priority_order = ["human_transfer", "wechat_invoke", "lead_guide"]

        for intent_type in priority_order:
            intent_cfg = nlu_config.get(intent_type, {})
            keywords = intent_cfg.get("keywords", [])

            for keyword in keywords:
                if keyword.lower() in message_lower:
                    logger.info("NLU matched intent=%s keyword=%s", intent_type, keyword)
                    return NLUIntent(
                        intent=intent_type,
                        confidence=0.9,
                        metadata={
                            "matched_keyword": keyword,
                            "command": intent_cfg.get("command", ""),
                            "response_template": intent_cfg.get("response_template", ""),
                        },
                    )

        # 未命中任何意图 -> 普通问答
        return NLUIntent(intent="qa", confidence=1.0)

    def get_intent_response(self, intent: NLUIntent) -> Optional[str]:
        """获取意图对应的预设回复"""
        if intent.intent == "qa":
            return None
        return intent.metadata.get("response_template")

    def get_intent_command(self, intent: NLUIntent) -> Optional[str]:
        """获取意图对应的业务指令"""
        if intent.intent == "qa":
            return None
        return intent.metadata.get("command")

    def _merge_config(self, agent_nlu_config: Optional[Dict]) -> Dict:
        """合并Agent自定义NLU配置与默认配置"""
        if not agent_nlu_config:
            return self._default_config

        merged = {}
        for intent_type, default_cfg in self._default_config.items():
            agent_cfg = agent_nlu_config.get(intent_type, {})
            merged[intent_type] = {
                "keywords": agent_cfg.get("keywords", default_cfg["keywords"]),
                "response_template": agent_cfg.get("response_template", default_cfg["response_template"]),
                "command": agent_cfg.get("command", default_cfg["command"]),
            }
        return merged


# 全局单例
nlu_engine = NLUEngine()
