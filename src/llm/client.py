"""
AI Teacher Robot — LLM Client.

This module provides a client for communicating with a locally-hosted
Large Language Model (LLM) via HTTP API.

The client sends text prompts to the LLM and receives text responses.
It supports configurable timeouts, model selection, and request parameters.

Usage:
    from src.llm.client import LLMClient

    client = LLMClient(api_url="http://localhost:11434/api/chat")
    response = client.send("What is the capital of France?")
"""

from __future__ import annotations

import json
from typing import Any, Optional

import requests

from src.config.settings import Settings
from src.utils.constants import DEFAULT_LLM_API_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_TIMEOUT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """HTTP client for communicating with a local LLM API.

    This class wraps HTTP requests to a local LLM API (e.g., Ollama,
    llama.cpp server) and provides a simple interface for sending prompts
    and receiving responses.

    Attributes:
        api_url: The URL of the LLM API endpoint.
        model: The LLM model name to use.
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_LLM_API_URL,
        model: str = DEFAULT_LLM_MODEL,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_tokens: int = 500,
        temperature: float = 0.7,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the LLMClient.

        Args:
            api_url: The URL of the LLM API endpoint.
            model: The LLM model name to use.
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            api_url = settings.llm_api_url
            model = settings.llm_model
            timeout = settings.llm_timeout
            max_tokens = settings.llm_max_tokens
            temperature = settings.llm_temperature

        self.api_url: str = api_url
        self.model: str = model
        self.timeout: int = timeout
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

        # HTTP session for connection reuse
        self._session: requests.Session = requests.Session()

    def send(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Send a prompt to the LLM and return the response.

        Args:
            prompt: The user's input prompt.
            system_prompt: Optional system prompt to set the LLM's behavior.
            **kwargs: Additional parameters to pass to the API
                (e.g., temperature, max_tokens).

        Returns:
            The LLM's response text.

        Raises:
            ConnectionError: If the LLM API is not reachable.
            TimeoutError: If the request times out.
            RuntimeError: If the API returns an error.
        """
        logger.info(f"Sending prompt to LLM (model={self.model})...")

        # Build the request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        # Add system prompt if provided
        if system_prompt:
            payload["messages"].append(
                {"role": "system", "content": system_prompt}
            )

        # Add user prompt
        payload["messages"].append(
            {"role": "user", "content": prompt}
        )

        try:
            response: requests.Response = self._session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            logger.error(f"Failed to connect to LLM API: {exc}")
            raise ConnectionError(
                f"Cannot connect to LLM API at {self.api_url}. "
                "Ensure the LLM server is running."
            ) from exc
        except requests.exceptions.Timeout as exc:
            logger.error(f"LLM API request timed out: {exc}")
            raise TimeoutError(
                f"LLM API request timed out after {self.timeout} seconds."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            logger.error(f"LLM API returned an error: {exc}")
            raise RuntimeError(
                f"LLM API error: {response.status_code} - {response.text}"
            ) from exc

        # Parse the response
        result: dict[str, Any] = response.json()
        response_text: str = self._extract_response_text(result)

        logger.info(f"LLM response received ({len(response_text)} chars).")
        return response_text

    def _extract_response_text(self, result: dict[str, Any]) -> str:
        """Extract the response text from the API response.

        Different LLM APIs return responses in different formats.
        This method handles common formats (Ollama, llama.cpp, OpenAI-compatible).

        Args:
            result: The parsed JSON response from the API.

        Returns:
            The extracted response text.
        """
        # Ollama format: {"message": {"content": "..."}}
        if "message" in result and isinstance(result["message"], dict):
            return result["message"].get("content", "")

        # OpenAI-compatible format: {"choices": [{"message": {"content": "..."}}]}
        if "choices" in result and isinstance(result["choices"], list):
            if result["choices"]:
                choice = result["choices"][0]
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        return message.get("content", "")

        # llama.cpp format: {"content": "..."}
        if "content" in result:
            return result["content"]

        # Fallback: try to find any string field
        logger.warning(
            f"Unknown response format from LLM API: {result}"
        )
        return ""

    def is_available(self) -> bool:
        """Check if the LLM API is reachable.

        Returns:
            True if the API is reachable, False otherwise.
        """
        try:
            response: requests.Response = self._session.get(
                self.api_url.replace("/api/chat", "/api/tags"),
                timeout=5,
            )
            return response.status_code == 200
        except (requests.exceptions.RequestException, ValueError):
            return False

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        logger.info("Closing LLM client session...")
        self._session.close()
