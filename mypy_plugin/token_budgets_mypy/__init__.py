"""token_budgets_mypy: production-grade Mypy plugin for affine
Budget enforcement on token_budgets.Budget instances."""
from .plugin import plugin

__all__ = ["plugin"]
__version__ = "0.2.0"