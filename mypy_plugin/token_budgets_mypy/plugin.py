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

SKIP_FILES = {
    "token_budgets.py",
    "token_budgets/__init__.py",
}

TB_DOUBLE_USE_ERROR = ErrorCode(
    code="tb-double-use",
    description="Affine ownership violation on Budget",
    category="general",
)

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

    if recv_name == "self":
        return ctx.default_return_type

    file = _current_file(ctx)

    if file in SKIP_FILES:
        return ctx.default_return_type

    consumed = _CONSUMED.setdefault(file, {})

    if recv_name in consumed and method_name not in NON_CONSUMING_METHODS:
        ctx.api.fail(
            f"Budget '{recv_name}' was already consumed by "
            f"{consumed[recv_name]}; using it again here violates "
            f"affine ownership.",
            ctx.context,
            code=TB_DOUBLE_USE_ERROR,
        )

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
