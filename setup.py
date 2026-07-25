"""
setup.py — Installation script for the AI Teacher Robot package.

This file allows the project to be installed as a Python package using:
    pip install -e .

It reads metadata from this file and (optionally) from pyproject.toml.
"""

from setuptools import find_packages, setup

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from src/__init__.py or a dedicated version file
# (Placeholder — will be updated when the first version is released)
VERSION = "0.1.0"

setup(
    name="ai-teacher-robot",
    version=VERSION,
    author="Bilal Tariq",
    author_email="bilaltariq221744-png@users.noreply.github.com",
    description=(
        "An AI-powered educational robot for Grade 2-10 students, "
        "running locally on a Raspberry Pi with multilingual voice interaction."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bilaltariq221744-png/AI_Teacher_Robot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        # Runtime dependencies are listed in requirements.txt.
        # This list is kept in sync with requirements.txt.
        "PyYAML>=6.0,<7.0",
        "loguru>=0.7,<0.8",
        "python-dotenv>=1.0,<2.0",
        "requests>=2.31,<3.0",
        "python-dateutil>=2.8,<3.0",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "ai-teacher-robot=src.main:main",
        ],
    },
)
