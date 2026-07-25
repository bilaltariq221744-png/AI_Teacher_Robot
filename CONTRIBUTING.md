# Contributing to AI Teacher Robot

First off, thank you for taking the time to contribute! 🎉

This document outlines the guidelines and conventions for contributing to the
**AI Teacher Robot** project. By contributing, you agree that your submissions
will be licensed under the [MIT License](LICENSE).

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Branch Naming Convention](#branch-naming-convention)
4. [Coding Standards](#coding-standards)
5. [Commit Message Convention](#commit-message-convention)
6. [Pull Request Process](#pull-request-process)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Project Structure](#project-structure)
10. [Reporting Issues](#reporting-issues)

---

## Code of Conduct

We expect all contributors to uphold a respectful and inclusive environment.
Please be kind, constructive, and collaborative. Harassment or discrimination
of any kind will not be tolerated.

---

## Getting Started

### Prerequisites

- **Python** 3.11 or higher
- **Raspberry Pi OS** (64-bit recommended) — for hardware testing
- **Git**

### Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/AI_Teacher_Robot.git
   cd AI_Teacher_Robot
   ```

3. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. Install pre-commit hooks (optional but recommended):

   ```bash
   pre-commit install
   ```

---

## Branch Naming Convention

Use descriptive branch names that follow this pattern:

```
<type>/<short-description>
```

| Type | Purpose |
|---|---|
| `feature/` | New feature or functionality |
| `fix/` | Bug fix |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring without behavior change |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance, tooling, or dependency updates |

**Examples:**

```
feature/multilingual-stt
fix/urdu-tts-crash
docs/update-srs
refactor/llm-client
test/hardware-animations
chore/update-requirements
```

---

## Coding Standards

### Python Style Guide

- Follow **PEP 8** — the official Python style guide.
- Use **4 spaces** for indentation (no tabs).
- Maximum line length: **88 characters** (enforced by `black` and `ruff`).
- Use **type hints** for all function and method signatures.
- Write **docstrings** for every module, class, and public function using
  **Google-style** docstrings.

### Example Module Structure

```python
"""
Module docstring describing the purpose of this module.

This module provides utilities for...
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ExampleClass:
    """Brief description of the class.

    Detailed description of the class, its responsibilities, and usage.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the ExampleClass.

        Args:
            config: Optional configuration dictionary.
        """
        self._config = config or {}

    def do_something(self, value: str) -> str:
        """Perform an action with the given value.

        Args:
            value: The input string to process.

        Returns:
            The processed string.

        Raises:
            ValueError: If the value is empty.
        """
        if not value:
            raise ValueError("value must not be empty")
        return value.strip()
```

### Import Ordering

Imports must be sorted using `isort` and follow this order:

1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Optional

# Third-party
import requests
import yaml

# Local
from src.config import settings
from src.utils.logger import get_logger
```

### Linting and Formatting

Before submitting a pull request, ensure your code passes all checks:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Run tests
pytest tests/ -v --cov=src
```

---

## Commit Message Convention

We follow the **[Conventional Commits](https://www.conventionalcommits.org/)**
specification:

```
<type>(<scope>): <subject>

<body>
```

| Type | When to use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, whitespace, etc.) |
| `refactor` | Code refactoring without behavior change |
| `perf` | Performance improvements |
| `test` | Adding or fixing tests |
| `chore` | Maintenance, tooling, or dependency updates |
| `build` | Changes affecting the build system or dependencies |
| `ci` | Changes to CI configuration |

**Example:**

```
feat(speech): add Urdu language detection module

Added LanguageDetector class that identifies whether the input
audio is in English or Urdu using a lightweight statistical model.

Closes #42
```

---

## Pull Request Process

1. Ensure your branch is up to date with `main`:

   ```bash
   git checkout main
   git pull origin main
   git checkout <your-branch>
   git rebase main
   ```

2. Ensure all tests pass:

   ```bash
   pytest tests/ -v --cov=src
   ```

3. Ensure code quality checks pass:

   ```bash
   ruff check src/ tests/
   mypy src/
   ```

4. Open a Pull Request on GitHub.

5. Fill in the PR template with:
   - A clear description of the changes
   - Related issue numbers (e.g., `Closes #42`)
   - Any breaking changes

6. A maintainer will review your PR. Please address all feedback promptly.

---

## Testing

- All new features and bug fixes **must** include tests.
- Use `pytest` as the test framework.
- Place tests in the `tests/` directory, mirroring the `src/` structure.
- Use `pytest-mock` for mocking external dependencies.
- Run tests with coverage:

  ```bash
  pytest tests/ -v --cov=src --cov-report=html
  ```

---

## Documentation

- Update documentation in `docs/` whenever you add or change functionality.
- Keep the [SRS](docs/SRS.md), [Architecture](docs/Architecture.md), and
  [Research](docs/Research.md) documents up to date.
- Use clear, concise language and include diagrams where helpful.

---

## Project Structure

Please follow the existing project structure. Do not create new top-level
directories without discussing with the team first.

```
src/
├── speech/    # Speech-to-text & language detection
├── llm/       # LLM client & prompt management
├── tts/       # Text-to-speech synthesis
├── hardware/  # Robot animation & servo control
├── config/    # Configuration management
└── utils/     # Shared utilities & helpers
```

---

## Reporting Issues

If you find a bug or have a feature request, please
[open an issue](https://github.com/bilaltariq221744-png/AI_Teacher_Robot/issues)
on GitHub. Include:

- A clear title and description
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Your environment (Python version, Raspberry Pi model, OS)
- Any relevant logs or error messages

---

Thank you for contributing to the AI Teacher Robot! 🚀
