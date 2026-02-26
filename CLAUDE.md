# modded-nanoLOOP

Looped Transformer speedrun — depth-recurrent GPT-2 on FineWeb.

## Philosophy
Research code — simple, correct, and efficient:
- Simple, hackable implementations > frameworks
- Correctness is non-negotiable — write pytest tests for non-trivial functions
- Tests should cover behavior and edge cases, not implementation details — keep them maintainable so refactors don't require rewriting every test
- Compute is scarce (2×A100-SXM4-80GB) — always consider memory, FLOPs, and throughput implications

## Code Standards
- Before writing new functions, check existing modules for code that can be extended or reused
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)

## Package Management (CRITICAL)
- ALWAYS: `uv add <package>`
- NEVER: manually edit pyproject.toml
- NEVER: `pip install` or `uv pip install`

## Running Code
Python scripts must be run within the uv environment

## Debugging
Check `.venv` source code directly for library implementation details