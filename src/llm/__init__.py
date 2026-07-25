"""
AI Teacher Robot — LLM Package.

This package handles communication with the local Large Language Model (LLM),
including sending prompts, receiving responses, and managing prompt templates.

Usage:
    from src.llm.client import LLMClient
    from src.llm.prompt_manager import PromptManager
    from src.llm.response_handler import ResponseHandler
"""

from src.llm.client import LLMClient
from src.llm.prompt_manager import PromptManager
from src.llm.response_handler import ResponseHandler

__all__ = ["LLMClient", "PromptManager", "ResponseHandler"]
