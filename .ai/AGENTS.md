# AGENTS.md

This file provides guidance to code agents when working with code in this repository.

## Repository Overview

This is the codebase for **imggrep**, a cli tool for recursively searching directories for image similarity.

## Key Commands

### Code Formatting
```bash
# Format Python code with Ruff
conda activate imggrep
ruff check --fix --line-length 88 imggrep/ && ruff format --line-length 88 imggrep/
```
