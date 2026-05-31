# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None):

        self.model = model
        self.temperature = float(temperature) if temperature is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.max_tokens = int(max_tokens) if max_tokens is not None else None

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate model output from a given prompt."""
        raise NotImplementedError

