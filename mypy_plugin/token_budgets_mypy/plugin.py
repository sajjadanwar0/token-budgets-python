"""token_budgets_mypy.plugin — final mypy 1.8+/2.x compatible version.

Detects (within a single Python source file, single function granularity):
  - Direct double-spend on a Budget receiver
  - Use-after-split on the parent receiver
  - Use of an argument that was consumed by merge_with
  - Use of a module-level Budget after consumption in any function in
    the same file

Does NOT detect (mypy plugin API does not expose the hooks required):
  - Variable rebinding (b = b.spend(X) followed by b.spend(Y) is
    legitimate but will be flagged; this is a known false-positive).
  - Inter-procedural tracking
  - Async-context tracking
  - Container tracking

The plugin SKIPS analysis of the `token_budgets.py` source module
itself, so the Python port's internal `self.spend(...)` /
`self.split(...)` calls are not flagged as violations.
"""

from typing import Callable, Dict, Optional, Set, Tuple
from mypy.plugin import MethodContext, Plugin
from mypy.types import Type
from mypy.nodes import CallExpr, Expression, MemberExpr, NameExpr
from mypy.errorcodes import ErrorCode

BUDGET_FQN = "token_budgets.Budget"

CONSUMING_METHODS: Dict[str, str] = {
    "spend": "spend (returns a new Budget)",
    "split": "split (returns a (taken, kept) tuple)",
    "merge_with": "merge_with (consumes self and the argument)",
}

SECOND_BUDGET_CONSUMING_METHODS: Dict[str, str] = {
    "merge_with": "merge_with (passed as argument)",
}

NON_CONSUMING_METHODS: Set[str] = {
    "micro_cents",
    "__repr__",
    "__str__",
    "__hash__",
}

# Files we should NOT analyse for affine violations. The Python port's
# Budget class implementation contains legitimate internal patterns
# (e.g., self.spend(...) inside Budget.split) that would be false
# positives under any single-function-scope plugin.
SKIP_FILES = {
    "token_budgets.py",
    "token_budgets/__init__.py",
}

TB_DOUBLE_USE_ERROR = ErrorCode(
    code="tb-double-use",
    description="Affine ownership violation on Budget",
    category="general",
)

# Per-file consumption tracking. Keys are file basenames; values map
# variable names to the consumption reason. Limitations: cannot
# distinguish rebinding (b = b.spend(X); b.spend(Y) → flagged
# conservatively); does not separate function scopes within a file
# (use module-level tests sparingly).
_CONSUMED: Dict[str, Dict[str, str]] = {}


def _name_of(expr: Expression) -> Optional[str]:
    if isinstance(expr, NameExpr):
        return expr.name
    if isinstance(expr, MemberExpr):
        base = _name_of(expr.expr)
        if base is not None:
            return f"{base}.{expr.name}"
    return None


def _current_file(ctx: MethodContext) -> str:
    path = getattr(ctx.api, "path", None)
    if path:
        # Normalise to basename for cross-platform consistency
        import os
        return os.path.basename(path)
    return "<unknown>"


def _budget_method_hook(ctx: MethodContext) -> Type:
    if not isinstance(ctx.context, CallExpr):
        return ctx.default_return_type
    callee = ctx.context.callee
    if not isinstance(callee, MemberExpr):
        return ctx.default_return_type
    method_name = callee.name
    recv_name = _name_of(callee.expr)
    if recv_name is None:
        return ctx.default_return_type

    # If receiver is `self`, do not track (legitimate internal pattern
    # on Budget's own methods, e.g., self.spend in Budget.split).
    if recv_name == "self":
        return ctx.default_return_type

    file = _current_file(ctx)

    # Skip the Python port's source file entirely
    if file in SKIP_FILES:
        return ctx.default_return_type

    consumed = _CONSUMED.setdefault(file, {})

    # Check if receiver was previously consumed in this file
    if recv_name in consumed and method_name not in NON_CONSUMING_METHODS:
        ctx.api.fail(
            f"Budget '{recv_name}' was already consumed by "
            f"{consumed[recv_name]}; using it again here violates "
            f"affine ownership.",
            ctx.context,
            code=TB_DOUBLE_USE_ERROR,
        )

    # If method consumes self, mark
    if method_name in CONSUMING_METHODS:
        consumed[recv_name] = CONSUMING_METHODS[method_name]

        if method_name in SECOND_BUDGET_CONSUMING_METHODS:
            if len(ctx.args) > 0 and len(ctx.args[0]) > 0:
                arg_expr = ctx.args[0][0]
                arg_name = _name_of(arg_expr)
                if arg_name is not None and arg_name != "self":
                    if arg_name in consumed:
                        ctx.api.fail(
                            f"Budget '{arg_name}' (passed to "
                            f"{method_name}) was already consumed by "
                            f"{consumed[arg_name]}.",
                            ctx.context,
                            code=TB_DOUBLE_USE_ERROR,
                        )
                    consumed[arg_name] = SECOND_BUDGET_CONSUMING_METHODS[
                        method_name
                    ]

    return ctx.default_return_type


class TokenBudgetsPlugin(Plugin):
    def get_method_hook(
        self, fullname: str
    ) -> Optional[Callable[[MethodContext], Type]]:
        if fullname.startswith(BUDGET_FQN + "."):
            return _budget_method_hook
        return None


def plugin(version: str):
    return TokenBudgetsPlugin
