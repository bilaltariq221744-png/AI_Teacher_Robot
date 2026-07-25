# Development Guide

This guide explains how to set up, build, test, and contribute to the
**AI Teacher Robot** project. It is written to be practical and accessible
for both experienced developers and interns.

---

# Git Workflow and Contribution Process

## Overview

This project follows a professional Git workflow to maintain code quality,
collaboration and stability.

The workflow structure:

```
main
 |
 |
development
 |
 |
feature branches
```

This means:

- **`main`** holds stable, production-ready code.
- **`development`** is where active development and integration happens.
- **Feature branches** are short-lived branches created from `development`
  for each individual feature or task.

## Main Branch

- The `main` branch contains stable and reviewed code.
- The `main` branch represents production-ready code.
- Developers must **not** directly push code to the `main` branch.
- Code is merged into `main` only after proper review, testing, and approval.
- Only stable and tested versions should exist on `main`.

## Development Branch

- The `development` branch is the active development and integration branch.
- All new features and improvements are first merged into the `development`
  branch.
- Developers should test their changes in the `development` branch before
  a production release.
- The `development` branch contains the latest working version of the
  project.
- Developers and interns should normally work from the `development` branch.

## Feature Development Process

### Step 1: Create a Feature Branch

Developers should create a separate feature branch from `development`.

```bash
git checkout development

git pull

git checkout -b feature/feature-name
```

Examples:

```
feature/speech-to-text
feature/llm-integration
feature/tts-module
feature/hardware-control
feature/raspberry-pi-deployment
```

### Step 2: Implement the Assigned Feature

- Work only on your assigned feature.
- Follow the existing project architecture.
- Write clean and modular code.
- Add comments where required.
- Update documentation if the architecture changes.
- Avoid unnecessary changes outside the assigned task.

### Step 3: Test the Feature

- Test the feature locally.
- Ensure existing functionality is not broken.
- Add or update tests if required.

### Step 4: Commit and Push Feature Branch

```bash
git add .

git commit -m "Add feature description"

git push origin feature/feature-name
```

Use meaningful commit messages.

**Good examples:**

```
Add Whisper speech recognition module
Implement Raspberry Pi servo controller
Improve Urdu language prompt handling
```

**Bad examples:**

```
update code
changes
fix
```

### Step 5: Create Pull Request

The developer creates a Pull Request:

```
feature branch → development branch
```

The Pull Request should contain:

- Description of changes
- Testing performed
- Related issue/task
- Screenshots/logs if required
- Documentation updates

### Step 6: Code Review

The team lead reviews:

- Code quality
- Functionality
- Testing results
- Compatibility with project architecture
- Documentation updates

### Step 7: Merge into Development

After approval:

```
feature branch → development branch
```

The feature becomes part of the active development version.

## Production Release Process

```
development branch
    |
    |
    |  Testing and final review
    |
    |
main branch
```

- Features are first tested in `development`.
- Only stable, tested, and approved changes are merged into `main`.
- The `main` branch should always contain reliable code.
- Production deployment should use the `main` branch.

## Important Rules for Developers and Interns

- Never directly push code to `main`.
- Never directly push unfinished features.
- Always create a feature branch before development.
- One feature should have one dedicated branch.
- Always pull the latest `development` branch before starting work.
- Write meaningful commit messages.
- Keep commits small and understandable.
- Update documentation when adding major features.
- Communicate major architectural changes before implementation.
