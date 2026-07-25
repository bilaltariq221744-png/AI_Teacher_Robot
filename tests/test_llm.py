"""
AI Teacher Robot — Tests for the LLM Module.

This module contains unit tests for the LLM-related components:
    - LLMClient
    - PromptManager
    - ResponseHandler

Usage:
    pytest tests/test_llm.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import LLMClient
from src.llm.prompt_manager import PromptManager
from src.llm.response_handler import ResponseHandler
from src.utils.constants import LANGUAGE_ENGLISH, LANGUAGE_URDU


class TestLLMClient:
    """Tests for the LLMClient class."""

    def test_init_default(self, mock_settings):
        """Test LLMClient initialization with default parameters."""
        client = LLMClient(settings=mock_settings)
        assert client.api_url == "http://localhost:11434/api/chat"
        assert client.model == "llama3"
        assert client.timeout == 30
        assert client.max_tokens == 500
        assert client.temperature == 0.7

    def test_init_custom(self):
        """Test LLMClient initialization with custom parameters."""
        client = LLMClient(
            api_url="http://localhost:8080/api/chat",
            model="gemma",
            timeout=60,
            max_tokens=1000,
            temperature=0.5,
        )
        assert client.api_url == "http://localhost:8080/api/chat"
        assert client.model == "gemma"
        assert client.timeout == 60
        assert client.max_tokens == 1000
        assert client.temperature == 0.5

    def test_extract_response_ollama(self):
        """Test _extract_response_text() with Ollama format."""
        client = LLMClient()
        result = client._extract_response_text(
            {"message": {"content": "Hello, world!"}}
        )
        assert result == "Hello, world!"

    def test_extract_response_openai(self):
        """Test _extract_response_text() with OpenAI-compatible format."""
        client = LLMClient()
        result = client._extract_response_text(
            {"choices": [{"message": {"content": "Hello!"}}]}
        )
        assert result == "Hello!"

    def test_extract_response_llamacpp(self):
        """Test _extract_response_text() with llama.cpp format."""
        client = LLMClient()
        result = client._extract_response_text({"content": "Hello!"})
        assert result == "Hello!"

    def test_extract_response_unknown_format(self):
        """Test _extract_response_text() with unknown format."""
        client = LLMClient()
        result = client._extract_response_text({"unknown": "data"})
        assert result == ""

    def test_close(self, mock_settings):
        """Test close() method."""
        client = LLMClient(settings=mock_settings)
        client.close()  # Should not raise


class TestPromptManager:
    """Tests for the PromptManager class."""

    def test_init_default(self):
        """Test PromptManager initialization with default parameters."""
        manager = PromptManager()
        assert manager.default_language == LANGUAGE_ENGLISH
        assert "helpful AI teacher" in manager.system_prompt_en
        assert "مددگار" in manager.system_prompt_ur

    def test_get_system_prompt_en(self):
        """Test get_system_prompt() for English."""
        manager = PromptManager()
        prompt = manager.get_system_prompt("en")
        assert "helpful AI teacher" in prompt

    def test_get_system_prompt_ur(self):
        """Test get_system_prompt() for Urdu."""
        manager = PromptManager()
        prompt = manager.get_system_prompt("ur")
        assert "مددگار" in prompt

    def test_build_prompt(self):
        """Test build_prompt() method."""
        manager = PromptManager()
        prompt = manager.build_prompt(
            "What is photosynthesis?", language="en"
        )
        assert prompt == "What is photosynthesis?"

    def test_build_prompt_with_context(self):
        """Test build_prompt() with context."""
        manager = PromptManager()
        prompt = manager.build_prompt(
            "What is photosynthesis?",
            language="en",
            context="Previous question: What is a plant?",
        )
        assert "Previous question" in prompt
        assert "photosynthesis" in prompt

    def test_set_system_prompt(self):
        """Test set_system_prompt() method."""
        manager = PromptManager()
        manager.set_system_prompt("en", "New prompt")
        assert manager.system_prompt_en == "New prompt"

    def test_set_system_prompt_urdu(self):
        """Test set_system_prompt() for Urdu."""
        manager = PromptManager()
        manager.set_system_prompt("ur", "نیا پرومنٹ")
        assert manager.system_prompt_ur == "نیا پرومنٹ"


class TestResponseHandler:
    """Tests for the ResponseHandler class."""

    def test_init_default(self):
        """Test ResponseHandler initialization with default parameters."""
        handler = ResponseHandler()
        assert handler.max_length == 500
        assert handler.remove_markdown is True
        assert handler.remove_code_blocks is True

    def test_process_simple_text(self):
        """Test process() with simple text."""
        handler = ResponseHandler()
        result = handler.process("Hello, world!")
        assert result == "Hello, world!"

    def test_process_markdown_bold(self):
        """Test process() removes Markdown bold formatting."""
        handler = ResponseHandler()
        result = handler.process("**Hello** world")
        assert result == "Hello world"

    def test_process_code_block(self):
        """Test process() removes code blocks."""
        handler = ResponseHandler()
        result = handler.process("```python\nprint('hello')\n```\nText")
        assert "print" not in result
        assert "Text" in result

    def test_process_empty(self):
        """Test process() with empty text."""
        handler = ResponseHandler()
        result = handler.process("")
        assert result == ""

    def test_process_truncation(self):
        """Test process() truncates long text."""
        handler = ResponseHandler(max_length=10)
        result = handler.process("This is a very long text")
        assert len(result) <= 10

    def test_validate_valid(self):
        """Test validate() with valid text."""
        handler = ResponseHandler()
        assert handler.validate("Hello") is True

    def test_validate_empty(self):
        """Test validate() with empty text."""
        handler = ResponseHandler()
        assert handler.validate("") is False

    def test_validate_too_long(self):
        """Test validate() with text exceeding max length."""
        handler = ResponseHandler(max_length=5)
        assert handler.validate("This is too long") is False

    def test_process_and_validate_valid(self):
        """Test process_and_validate() with valid text."""
        handler = ResponseHandler()
        result = handler.process_and_validate("Hello, world!")
        assert result == "Hello, world!"

    def test_process_and_validate_invalid(self):
        """Test process_and_validate() with invalid (empty) text."""
        handler = ResponseHandler()
        result = handler.process_and_validate("")
        assert result is None
