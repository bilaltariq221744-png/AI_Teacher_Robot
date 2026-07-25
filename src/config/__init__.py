"""
AI Teacher Robot — Configuration Package.

This package manages all application configuration, including loading settings
from YAML files and environment variables, and defining project file paths.

Usage:
    from src.config.settings import Settings

    settings = Settings()
    print(settings.llm_api_url)
"""

from src.config.settings import Settings

__all__ = ["Settings"]
