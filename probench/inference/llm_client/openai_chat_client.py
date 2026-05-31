# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from openai import OpenAI
from probench.inference.llm_client.base import BaseLLMClient

class OpenAIChatCompletionsClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
        model: str,
        temperature=None,
        top_p=None,
        max_tokens=None
    ):
        super().__init__(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.client = OpenAI(
            base_url=endpoint_url,
            api_key=api_key)

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

        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content

