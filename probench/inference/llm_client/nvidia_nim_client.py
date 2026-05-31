# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

import json
import requests
from probench.inference.llm_client.base import BaseLLMClient


class NvidiaNIMClient(BaseLLMClient):
    """Client for NVIDIA NIM inference API (OpenAI-compatible)."""

    ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

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
        self.api_key = api_key
        self.thinking = thinking

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        payload["chat_template_kwargs"] = {"thinking": self.thinking}

        response = requests.post(self.ENDPOINT, headers=headers, json=payload, timeout=600, stream=True)
        response.raise_for_status()

        chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

        return "".join(chunks)
