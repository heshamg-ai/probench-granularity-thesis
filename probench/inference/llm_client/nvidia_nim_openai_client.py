# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from openai import OpenAI
from probench.inference.llm_client.base import BaseLLMClient


class NvidiaNIMOpenAIClient(BaseLLMClient):
    """NVIDIA NIM client using the OpenAI SDK (for models like z-ai/glm5).

    Supports extra_body for model-specific chat_template_kwargs, e.g.:
      GLM-5: {"enable_thinking": False, "clear_thinking": True}
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking_kwargs: dict | None = None,
    ):
        super().__init__(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key)
        self.thinking_kwargs = thinking_kwargs or {}

    def generate(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.thinking_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": self.thinking_kwargs}

        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
