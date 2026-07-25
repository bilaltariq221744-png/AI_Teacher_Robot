"""
AI Teacher Robot — LLM Response Handler.

This module processes and validates LLM responses before passing them to the
TTS module. It handles tasks such as cleaning up the response text, removing
unwanted formatting, and enforcing length limits.

Usage:
    from src.llm.response_handler import ResponseHandler

    handler = ResponseHandler(max_length=500)
    clean_text = handler.process(raw_response)
"""

from __future__ import annotations

import re
from typing import Optional

from src.config.settings import Settings
from src.utils.constants import MAX_RESPONSE_LENGTH
from src.utils.helpers import truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseHandler:
    """Processes and validates LLM responses.

    This class cleans up LLM responses by removing unwanted formatting,
    enforcing length limits, and validating the content before it is
    passed to the TTS module.

    Attributes:
        max_length: Maximum number of characters in the processed response.
        remove_markdown: Whether to strip Markdown formatting.
        remove_code_blocks: Whether to strip code blocks.
    """

    # Patterns for cleaning up responses
    _MARKDOWN_PATTERNS: list[tuple[str, str]] = [
        (r"\*\*(.*?)\*\*", r"\1"),  # Bold: **text** -> text
        (r"\*(.*?)\*", r"\1"),      # Italic: *text* -> text
        (r"__(.*?)__", r"\1"),      # Bold: __text__ -> text
        (r"_(.*?)_", r"\1"),        # Italic: _text_ -> text
        (r"~~(.*?)~~", r"\1"),      # Strikethrough: ~~text~~ -> text
    ]

    _CODE_BLOCK_PATTERN: str = r"```[\s\S]*?```"
    _INLINE_CODE_PATTERN: str = r"`([^`]+)`"
    _MULTIPLE_NEWLINES: str = r"\n{3,}"
    _MULTIPLE_SPACES: str = r" {2,}"

    def __init__(
        self,
        max_length: int = MAX_RESPONSE_LENGTH,
        remove_markdown: bool = True,
        remove_code_blocks: bool = True,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the ResponseHandler.

        Args:
            max_length: Maximum number of characters in the processed response.
            remove_markdown: Whether to strip Markdown formatting.
            remove_code_blocks: Whether to strip code blocks.
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            max_length = settings.llm_max_tokens

        self.max_length: int = max_length
        self.remove_markdown: bool = remove_markdown
        self.remove_code_blocks: bool = remove_code_blocks

    def process(self, text: str) -> str:
        """Process and clean an LLM response.

        This method performs the following steps:
            1. Strips leading/trailing whitespace.
            2. Removes code blocks (if enabled).
            3. Removes Markdown formatting (if enabled).
            4. Normalizes whitespace.
            5. Truncates to the maximum length.

        Args:
            text: The raw LLM response text.

        Returns:
            The cleaned and validated response text.
        """
        if not text:
            logger.warning("Empty response received from LLM.")
            return ""

        logger.debug(f"Processing LLM response ({len(text)} chars)...")

        processed: str = text.strip()

        # Remove code blocks
        if self.remove_code_blocks:
            processed = re.sub(self._CODE_BLOCK_PATTERN, "", processed)

        # Remove inline code (keep the content)
        processed = re.sub(self._INLINE_CODE_PATTERN, r"\1", processed)

        # Remove Markdown formatting
        if self.remove_markdown:
            for pattern, replacement in self._MARKDOWN_PATTERNS:
                processed = re.sub(pattern, replacement, processed)

        # Normalize whitespace
        processed = re.sub(self._MULTIPLE_NEWLINES, "\n\n", processed)
        processed = re.sub(self._MULTIPLE_SPACES, " ", processed)
        processed = processed.strip()

        # Truncate to maximum length
        processed = truncate_text(processed, self.max_length)

        logger.debug(f"Processed response ({len(processed)} chars).")
        return processed

    def validate(self, text: str) -> bool:
        """Validate an LLM response.

        Checks if the response is non-empty and within the length limit.

        Args:
            text: The response text to validate.

        Returns:
            True if the response is valid, False otherwise.
        """
        if not text or not text.strip():
            return False

        if len(text) > self.max_length:
            logger.warning(
                f"Response exceeds max length ({len(text)} > {self.max_length})."
            )
            return False

        return True

    def process_and_validate(self, text: str) -> Optional[str]:
        """Process and validate an LLM response.

        This is a convenience method that processes the response and then
        validates it. If the response is invalid after processing, None is
        returned.

        Args:
            text: The raw LLM response text.

        Returns:
            The processed and validated response text, or None if invalid.
        """
        processed: str = self.process(text)

        if self.validate(processed):
            return processed

        logger.warning("Response failed validation after processing.")
        return None
