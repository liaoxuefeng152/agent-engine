"""
多模型智能路由与降级服务
- 根据Agent配置选择主模型/备用模型
- 模型失败自动切换备用模型
- 模型超时熔断机制
"""
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

from coze_coding_dev_sdk import LLMClient
from coze_coding_utils.runtime_ctx.context import new_context
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


@dataclass
class ModelRouteConfig:
    """模型路由配置"""
    primary: str = "doubao-seed-2-0-lite-260215"
    fallback: List[str] = field(default_factory=lambda: ["doubao-seed-2-0-mini-260215"])
    timeout: int = 30
    max_retries: int = 2


@dataclass
class ModelCallResult:
    """模型调用结果"""
    content: str = ""
    model_used: str = ""
    token_input: int = 0
    token_output: int = 0
    success: bool = True
    error: Optional[str] = None


class CircuitBreaker:
    """熔断器 - 防止故障模型阻塞"""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self._failure_counts: Dict[str, int] = {}
        self._open_until: Dict[str, float] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

    def is_open(self, model: str) -> bool:
        """检查熔断器是否打开(不可用)"""
        until = self._open_until.get(model, 0)
        if until > time.time():
            return True
        if until > 0 and time.time() >= until:
            self._failure_counts[model] = 0
            self._open_until[model] = 0
        return False

    def record_success(self, model: str):
        self._failure_counts[model] = 0
        self._open_until[model] = 0

    def record_failure(self, model: str):
        count = self._failure_counts.get(model, 0) + 1
        self._failure_counts[model] = count
        if count >= self._failure_threshold:
            self._open_until[model] = time.time() + self._recovery_timeout
            logger.warning("Circuit breaker OPEN for model=%s", model)


class ModelRouter:
    """
    多模型智能路由器
    - 根据Agent配置选择模型
    - 自动降级到备用模型
    - 熔断器保护
    """

    def __init__(self):
        self._breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

    def _parse_route_config(self, model_config: dict) -> ModelRouteConfig:
        return ModelRouteConfig(
            primary=model_config.get("primary", "doubao-seed-2-0-lite-260215"),
            fallback=model_config.get("fallback", ["doubao-seed-2-0-mini-260215"]),
            timeout=model_config.get("timeout", 30),
            max_retries=model_config.get("max_retries", 2),
        )

    def _get_candidate_models(self, route_cfg: ModelRouteConfig) -> List[str]:
        candidates = []
        if not self._breaker.is_open(route_cfg.primary):
            candidates.append(route_cfg.primary)
        for m in route_cfg.fallback:
            if not self._breaker.is_open(m) and m not in candidates:
                candidates.append(m)
        if not candidates:
            candidates.append(route_cfg.primary)
            logger.warning("All models circuit-broken, forcing primary=%s", route_cfg.primary)
        return candidates

    def _sync_stream_call(self, messages: List[BaseMessage], model: str, temperature: float, max_tokens: int, ctx=None) -> list:
        """同步流式调用，收集所有chunk返回"""
        client = LLMClient(ctx=ctx)
        chunks = []
        usage_info = {}
        for chunk in client.stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        ):
            if chunk.content:
                chunks.append(chunk.content)
            if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                usage_info = chunk.response_metadata.get("usage", {})
        return chunks, usage_info

    async def call_stream(
        self,
        messages: List[BaseMessage],
        model_config: dict,
        ctx=None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[tuple]:
        """
        流式调用模型，自动降级
        yields: (chunk_content: str, is_final: bool, model_used: str, usage: dict)
        """
        route_cfg = self._parse_route_config(model_config)
        candidates = self._get_candidate_models(route_cfg)

        for model in candidates:
            try:
                logger.info("ModelRouter attempting model=%s", model)
                # 在线程池中执行同步流式调用
                chunks, usage_info = await asyncio.to_thread(
                    self._sync_stream_call, messages, model, temperature, max_tokens, ctx
                )

                # 逐chunk yield
                for chunk_content in chunks:
                    yield (chunk_content, False, model, {})

                self._breaker.record_success(model)
                yield ("", True, model, usage_info)
                return

            except asyncio.TimeoutError:
                logger.warning("ModelRouter model=%s timeout after %ds", model, route_cfg.timeout)
                self._breaker.record_failure(model)
            except Exception as e:
                logger.warning("ModelRouter model=%s failed: %s, trying next", model, e)
                self._breaker.record_failure(model)

        yield ("", True, "", {})
        logger.error("ModelRouter all models exhausted")

    async def call_invoke(
        self,
        messages: List[BaseMessage],
        model_config: dict,
        ctx=None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelCallResult:
        """非流式调用模型，自动降级"""
        route_cfg = self._parse_route_config(model_config)
        candidates = self._get_candidate_models(route_cfg)

        for model in candidates:
            try:
                client = LLMClient(ctx=ctx)
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.invoke,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    ),
                    timeout=route_cfg.timeout,
                )

                self._breaker.record_success(model)
                usage = response.response_metadata.get("usage", {}) if response.response_metadata else {}

                content = ""
                if isinstance(response.content, str):
                    content = response.content
                elif isinstance(response.content, list):
                    content = " ".join(
                        item.get("text", "") for item in response.content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ) or " ".join(str(x) for x in response.content)

                return ModelCallResult(
                    content=content,
                    model_used=model,
                    token_input=usage.get("prompt_tokens", 0),
                    token_output=usage.get("completion_tokens", 0),
                    success=True,
                )
            except asyncio.TimeoutError:
                logger.warning("ModelRouter model=%s timeout after %ds", model, route_cfg.timeout)
                self._breaker.record_failure(model)
            except Exception as e:
                logger.warning("ModelRouter model=%s failed: %s", model, e)
                self._breaker.record_failure(model)

        return ModelCallResult(success=False, error="All models exhausted")


# 全局单例
model_router = ModelRouter()
