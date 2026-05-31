# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from zai import ZhipuAiClient
from probench.inference.llm_client.base import BaseLLMClient


class ZhipuClient(BaseLLMClient):
    """Client for Zhipu AI (Z.ai) API using zai-sdk.

    Supports models like glm-5.1.
    Thinking is disabled by default.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
    ):
        super().__init__(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.client = ZhipuAiClient(api_key=api_key)
        self.thinking = thinking

    def generate(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
