"""
AI Teacher Robot — Prompt Manager.

This module manages prompt templates and builds prompts for the LLM based on
the student's input and the detected language.

Usage:
    from src.llm.prompt_manager import PromptManager

    manager = PromptManager()
    prompt = manager.build_prompt("What is photosynthesis?", language="en")
"""

from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.utils.constants import LANGUAGE_ENGLISH, LANGUAGE_URDU
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages prompt templates and builds prompts for the LLM.

    This class provides methods to build prompts that include the system
    prompt (based on language) and the student's question. It also
    supports prompt templates for different educational contexts.

    Attributes:
        system_prompt_en: The system prompt for English.
        system_prompt_ur: The system prompt for Urdu.
        default_language: The default language for prompts.
    """

    # Default system prompts
    DEFAULT_SYSTEM_PROMPT_EN: str = (
        "You are a helpful AI teacher for students in Grade 2 to Grade 10. "
        "Answer questions clearly and concisely. "
        "If the question is not educational, politely decline to answer."
    )

    DEFAULT_SYSTEM_PROMPT_UR: str = (
        "آپ ایک مددگار ای آئی اساتذہ ہیں جو گریڈ 2 سے گریڈ 10 تک کے طلباء کے لئے ہیں۔ "
        "سوالات کا واضح اور مختصر جواب دیں۔ "
        "اگر سوال تعلیمی نہ ہو تو آؤٹ کرنے سے انکار کریں۔"
    )

    def __init__(
        self,
        system_prompt_en: Optional[str] = None,
        system_prompt_ur: Optional[str] = None,
        default_language: str = LANGUAGE_ENGLISH,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the PromptManager.

        Args:
            system_prompt_en: The system prompt for English.
                If None, the default English prompt is used.
            system_prompt_ur: The system prompt for Urdu.
                If None, the default Urdu prompt is used.
            default_language: The default language for prompts.
            settings: Optional Settings instance for configuration.
        """
        if settings is not None:
            system_prompt_en = system_prompt_en or settings.system_prompt_en
            system_prompt_ur = system_prompt_ur or settings.system_prompt_ur
            default_language = settings.default_language

        self.system_prompt_en: str = (
            system_prompt_en or self.DEFAULT_SYSTEM_PROMPT_EN
        )
        self.system_prompt_ur: str = (
            system_prompt_ur or self.DEFAULT_SYSTEM_PROMPT_UR
        )
        self.default_language: str = default_language

    def get_system_prompt(self, language: str) -> str:
        """Get the system prompt for the specified language.

        Args:
            language: The language code (e.g., "en", "ur").

        Returns:
            The system prompt string for the specified language.
            Falls back to the English prompt if the language is not found.
        """
        if language == LANGUAGE_URDU:
            return self.system_prompt_ur
        return self.system_prompt_en

    def build_prompt(
        self,
        text: str,
        language: str,
        context: Optional[str] = None,
    ) -> str:
        """Build a prompt for the LLM.

        Combines the system prompt (based on language) with the student's
        question and optional context to form a complete prompt.

        Args:
            text: The student's question or input text.
            language: The language code (e.g., "en", "ur").
            context: Optional conversation context or additional instructions.

        Returns:
            The complete prompt string for the LLM.
        """
        logger.debug(
            f"Building prompt (language={language}, text={text[:50]!r}...)"
        )

        system_prompt: str = self.get_system_prompt(language)

        # TODO: Implement prompt template rendering
        #   This could use Jinja2 or a custom template engine.
        #   For now, the system prompt is returned separately and the
        #   LLM client handles combining it with the user prompt.

        # For now, return just the user text (system prompt is handled
        # by the LLM client separately)
        prompt: str = text

        if context:
            prompt = f"{context}\n\n{prompt}"

        return prompt

    def build_educational_prompt(
        self,
        subject: str,
        topic: str,
        question: str,
        language: str,
    ) -> str:
        """Build an educational prompt with subject and topic context.

        Args:
            subject: The subject (e.g., "Math", "Science").
            topic: The specific topic (e.g., "Fractions", "Photosynthesis").
            question: The student's question.
            language: The language code.

        Returns:
            The complete educational prompt string.
        """
        logger.debug(
            f"Building educational prompt (subject={subject}, topic={topic})"
        )

        # TODO: Implement educational prompt template
        #   This could include subject-specific instructions and context.

        return question

    def set_system_prompt(self, language: str, prompt: str) -> None:
        """Set the system prompt for a specific language.

        Args:
            language: The language code (e.g., "en", "ur").
            prompt: The new system prompt string.
        """
        logger.info(f"Setting system prompt for language: {language}")

        if language == LANGUAGE_URDU:
            self.system_prompt_ur = prompt
        else:
            self.system_prompt_en = prompt
