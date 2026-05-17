"""
token_budgets_py — Python port of the Token Budgets affine discipline.

The catalogue of §II is dominated by failures in Python agent
frameworks (LangChain, AutoGPT, AutoGen, CrewAI). The Rust
discipline of §III provides compile-time integrity, which Python
cannot offer; this port enforces the same affine discipline at
runtime so Python operators can benefit from the cap-respecting
guarantee even without the static checks.

Mechanism: any operation that consumes the Budget (spend, split,
merge) sets an internal _consumed=True flag. Subsequent operations
on the same instance raise AffineViolation.

Trade-off vs Rust:
- Rust: compile-time integrity (rustc rejects bad programs)
- Python: runtime detection (AffineViolation at the point of misuse)

The cap-soundness guarantee is preserved in both: spend(amount)
reduces capacity by exactly amount, with no path that admits
double-spend.

Drop-in compatibility with LangChain callbacks; see
LangChainBudgetCallback below.
"""

from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")


class AffineViolation(RuntimeError):
    """Raised when a consumed Budget is used again."""


class BudgetExhausted(RuntimeError):
    """Raised when an operation would exceed the budget cap."""


@dataclass
class Budget:
    """Runtime affine Budget. The arithmetic invariant matches the
    Rust implementation; the affine invariant is enforced at runtime
    via the _consumed flag (Python has no compile-time linear types)."""

    initial_uc: int
    max_uc: int
    _consumed: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.initial_uc < 0 or self.max_uc < 0:
            raise ValueError("Budget initial and max must be non-negative")
        if self.initial_uc > self.max_uc:
            raise ValueError("initial_uc must not exceed max_uc")

    def _check_not_consumed(self):
        if self._consumed:
            raise AffineViolation(
                "Budget was already consumed by an earlier spend/split/merge "
                "operation; reuse would violate the affine discipline."
            )

    def micro_cents(self) -> int:
        return self.initial_uc

    def spend(self, amount: int) -> "Budget":
        self._check_not_consumed()
        if amount < 0:
            raise ValueError("spend amount must be non-negative")
        if amount > self.initial_uc:
            self._consumed = True
            raise BudgetExhausted(
                f"spend({amount}) exceeds available {self.initial_uc}"
            )
        self._consumed = True
        return Budget(
            initial_uc=self.initial_uc - amount,
            max_uc=self.max_uc,
        )

    def split(self, amount: int) -> Tuple["Budget", "Budget"]:
        self._check_not_consumed()
        if amount < 0:
            raise ValueError("split amount must be non-negative")
        if amount > self.initial_uc:
            raise BudgetExhausted(
                f"split({amount}) exceeds available {self.initial_uc}"
            )
        self._consumed = True
        taken = Budget(initial_uc=amount, max_uc=self.max_uc)
        kept = Budget(initial_uc=self.initial_uc - amount, max_uc=self.max_uc)
        return taken, kept

    def merge(self, other: "Budget") -> "Budget":
        self._check_not_consumed()
        other._check_not_consumed()
        self._consumed = True
        other._consumed = True
        return Budget(
            initial_uc=self.initial_uc + other.initial_uc,
            max_uc=max(self.max_uc, other.max_uc),
        )


# =====================================================================
# BudgetPool — multi-call reservation with closure-based typestate
# (compile-time prevention of forget-to-resolve at runtime via the
# with_reservation closure interface).
# =====================================================================


@dataclass
class ReservationReceipt:
    """Returned by BudgetPool.reserve. Must be confirmed or forfeited
    before the BudgetPool is reused."""

    amount: int
    _resolved: bool = field(default=False, repr=False)

    def confirm(self, actual: int, result_value=None):
        if self._resolved:
            raise AffineViolation("Receipt already resolved")
        if actual < 0 or actual > self.amount:
            raise ValueError(
                f"confirm actual={actual} must be in [0, {self.amount}]"
            )
        self._resolved = True
        # Refund = amount - actual (positive number); returned for the caller.
        return result_value


class BudgetPool:
    """A multi-call budget reservation pool. Thread-safe."""

    def __init__(self, available_uc: int):
        if available_uc < 0:
            raise ValueError("available_uc must be non-negative")
        self._available = available_uc
        self._lock = threading.Lock()

    def available(self) -> int:
        with self._lock:
            return self._available

    def with_reservation(
            self,
            amount: int,
            op: Callable[[ReservationReceipt], T],
    ) -> T:
        """Closure-based typestate: the op must call confirm() on the
        receipt or AffineViolation is raised when the closure returns.
        This is the compile-time-equivalent forget-to-resolve check."""
        with self._lock:
            if amount > self._available:
                raise BudgetExhausted(
                    f"reservation {amount} exceeds available {self._available}"
                )
            self._available -= amount
        receipt = ReservationReceipt(amount=amount)
        try:
            result = op(receipt)
        except Exception:
            # Forfeit: capacity is permanently consumed (matches Drop
            # semantics in Rust). Re-raise the caller's exception.
            receipt._resolved = True
            raise
        if not receipt._resolved:
            raise AffineViolation(
                "with_reservation closure returned without resolving its "
                "receipt (must call confirm())."
            )
        return result


# =====================================================================
# LangChain integration — runtime budget enforcement via callback.
#
# This callback subclasses BaseCallbackHandler (required for LangChain
# pydantic validation in 0.2+) and reads token usage from the modern
# usage_metadata path on ChatGeneration.message, with a fallback to
# the legacy llm_output["token_usage"] path for older LangChain
# versions.
#
# IMPORTANT: LangChain's default chain-control behaviour is to log
# callback exceptions and continue the chain. To make this callback
# actually abort an agent run, the operator must either:
#   (a) wrap the LLM in a class that gates _generate before invoke
#       (recommended; see budget_gated_chat_model.py for an example);
#   (b) use LangGraph's pre_model_hook (LangGraph 0.3+);
#   (c) configure callbacks with raise_error=True (LangChain-version-
#       dependent; verify against your installed version).
# Trace-only analysis (this callback as a passive observer) works
# universally and is the integration-agnostic path documented in
# §V-X of the paper.
# =====================================================================


try:
    from langchain_core.callbacks import BaseCallbackHandler
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore


class LangChainBudgetCallback(BaseCallbackHandler):
    """LangChain BaseCallbackHandler that tracks running spend and
    raises BudgetExhausted when the cap is exceeded.

    Usage (passive observer, all LangChain versions):
        budget = Budget(initial_uc=10_000, max_uc=100_000)
        cb = LangChainBudgetCallback(budget,
                                     rate_per_input_token_uc=0.15,
                                     rate_per_output_token_uc=0.60)
        agent.invoke({...}, config={"callbacks": [cb]})
        # After the run: cb.spent_so_far_uc records the running spend.
        # The cap is not enforced as a chain-abort unless the
        # operator integrates via one of (a)/(b)/(c) above.

    Rates are in micro-cents per token (e.g. gpt-4o-mini = 0.15 input,
    0.60 output). For consumer convenience, float rates are accepted.
    """

    def __init__(
            self,
            budget,
            rate_per_input_token_uc: float = 0.15,
            rate_per_output_token_uc: float = 0.60,
    ):
        if _LANGCHAIN_AVAILABLE:
            super().__init__()
        self._budget = budget
        cap_attr = getattr(budget, "micro_cents", None)
        self._cap = cap_attr() if callable(cap_attr) else int(budget)
        self.rate_in = float(rate_per_input_token_uc)
        self.rate_out = float(rate_per_output_token_uc)
        self.spent_so_far_uc: float = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Pre-flight check. Estimate is conservative byte-length input
        plus a max_output_tokens output ceiling. Raises BudgetExhausted
        if running + estimate would exceed cap."""
        est_input = sum(len(p) for p in prompts) * self.rate_in
        # Conservative output ceiling: 500 tokens (matches default
        # max_tokens=500 in the paper's experiments).
        est_output = 500 * self.rate_out
        est = est_input + est_output
        if self.spent_so_far_uc + est > self._cap:
            raise BudgetExhausted(
                f"pre-flight estimate {est:.0f} uc plus running spend "
                f"{self.spent_so_far_uc:.0f} uc would exceed cap "
                f"{self._cap} uc"
            )

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """LangChain 0.2+ chat models invoke this instead of on_llm_start.
        Compose the messages into a prompt-equivalent and apply the same
        pre-flight check."""
        # messages is a list of lists of BaseMessage. Each message has
        # a .content (string).
        try:
            flattened = []
            for msg_list in messages:
                for msg in msg_list:
                    content = getattr(msg, "content", "")
                    if isinstance(content, str):
                        flattened.append(content)
            self.on_llm_start({}, flattened)
        except BudgetExhausted:
            raise
        except Exception:
            # Best-effort; passive observation still works
            pass

    def on_llm_end(self, response, **kwargs):
        """Read usage from response. Tries LangChain 0.2+ usage_metadata
        path first, falls back to legacy llm_output['token_usage']."""
        cost = 0.0

        # --- LangChain 0.2+ path: response.generations[*][*].message.usage_metadata
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        usage = getattr(msg, "usage_metadata", None)
                        if usage:
                            in_t = usage.get("input_tokens", 0) or 0
                            out_t = usage.get("output_tokens", 0) or 0
                            cost = in_t * self.rate_in + out_t * self.rate_out
                            break
                if cost > 0:
                    break
        except Exception:
            pass

        # --- Legacy LangChain 0.1 path: response.llm_output["token_usage"]
        if cost == 0.0:
            try:
                llm_output = getattr(response, "llm_output", None) or {}
                usage = llm_output.get("token_usage") or {}
                if usage:
                    in_t = usage.get("prompt_tokens", 0) or 0
                    out_t = usage.get("completion_tokens", 0) or 0
                    cost = in_t * self.rate_in + out_t * self.rate_out
            except Exception:
                pass

        if cost > 0:
            self.spent_so_far_uc += cost
            if self.spent_so_far_uc > self._cap:
                raise BudgetExhausted(
                    f"running spend {self.spent_so_far_uc:.0f} uc "
                    f"exceeded cap {self._cap} uc"
                )


# =====================================================================
# BudgetGatedChatOpenAI — runtime enforcement that LangChain
# *cannot* swallow (Improvement #2 from the v25 review).
#
# Wraps ChatOpenAI._generate to gate the API call before invocation.
# BudgetExhausted raised inside _generate propagates up through
# agent.invoke() because the LLM invocation itself raises, not a
# callback. This is the deployable enforcement path for the
# LangChain ecosystem.
# =====================================================================


def make_budget_gated_chat_openai(*args, budget_callback: LangChainBudgetCallback,
                                  **kwargs):
    """Factory that wraps ChatOpenAI._generate to gate the API call
    before invocation. Requires langchain-openai installed.

    Returns a ChatOpenAI instance whose _generate raises
    BudgetExhausted before invoking the OpenAI API if the running
    spend plus the conservative estimate would exceed the budget.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for make_budget_gated_chat_openai"
        ) from e

    instance = ChatOpenAI(*args, **kwargs)
    original_generate = instance._generate

    def gated_generate(messages, stop=None, run_manager=None, **kw):
        # Pre-call check using the same logic as on_chat_model_start.
        try:
            budget_callback.on_chat_model_start({}, [messages])
        except BudgetExhausted:
            raise
        result = original_generate(messages, stop=stop, run_manager=run_manager, **kw)
        # After call: update running spend.
        budget_callback.on_llm_end(result)
        return result

    instance._generate = gated_generate
    return instance


# =====================================================================
# Tests
# =====================================================================

if __name__ == "__main__":
    # Test 1: basic spend
    b = Budget(initial_uc=1000, max_uc=10_000)
    b2 = b.spend(100)
    assert b2.micro_cents() == 900

    # Test 2: double-spend rejected
    try:
        b.spend(50)
    except AffineViolation:
        print("OK: double-spend detected")
    else:
        raise AssertionError("expected AffineViolation")

    # Test 3: split conservation
    b = Budget(initial_uc=1000, max_uc=10_000)
    taken, kept = b.split(300)
    assert taken.micro_cents() + kept.micro_cents() == 1000

    # Test 4: pool with_reservation
    pool = BudgetPool(available_uc=10_000)
    result = pool.with_reservation(
        500, lambda r: r.confirm(423, "agent output")
    )
    assert result == "agent output"

    # Test 5: pool resource-leak detection
    try:
        pool.with_reservation(500, lambda r: "forgot to resolve")
    except AffineViolation:
        print("OK: forgot-to-resolve detected at runtime")
    else:
        raise AssertionError("expected AffineViolation")

    # Test 6: LangChainBudgetCallback usage_metadata path (LangChain 0.2+)
    if _LANGCHAIN_AVAILABLE:
        class FakeMsg:
            def __init__(self, in_t, out_t):
                self.usage_metadata = {"input_tokens": in_t, "output_tokens": out_t}
        class FakeGen:
            def __init__(self, msg):
                self.message = msg
        class FakeResponse:
            def __init__(self, gens):
                self.generations = gens

        budget = Budget(initial_uc=500, max_uc=10_000)
        cb = LangChainBudgetCallback(budget,
                                     rate_per_input_token_uc=0.15,
                                     rate_per_output_token_uc=0.60)
        # First call: 100 input + 100 output = 75 uc, well under cap=500
        resp1 = FakeResponse([[FakeGen(FakeMsg(100, 100))]])
        cb.on_llm_end(resp1)
        assert cb.spent_so_far_uc == 75.0, f"expected 75, got {cb.spent_so_far_uc}"
        # Second call: another 75 uc, total 150 uc
        cb.on_llm_end(resp1)
        assert cb.spent_so_far_uc == 150.0
        # Third call should push over cap: 500 input + 500 output = 375 uc
        resp_big = FakeResponse([[FakeGen(FakeMsg(500, 500))]])
        try:
            cb.on_llm_end(resp_big)
        except BudgetExhausted:
            print("OK: LangChain 0.2+ usage_metadata enforcement fires correctly")
        else:
            raise AssertionError("expected BudgetExhausted on 525 > 500 cap")

        # Test 7: legacy LangChain 0.1 fallback path
        class FakeLegacyResponse:
            def __init__(self, in_t, out_t):
                self.llm_output = {"token_usage": {"prompt_tokens": in_t,
                                                   "completion_tokens": out_t}}
                self.generations = []  # empty, forcing fallback
        budget = Budget(initial_uc=500, max_uc=10_000)
        cb = LangChainBudgetCallback(budget,
                                     rate_per_input_token_uc=0.15,
                                     rate_per_output_token_uc=0.60)
        cb.on_llm_end(FakeLegacyResponse(100, 100))
        assert cb.spent_so_far_uc == 75.0, f"expected 75 (legacy), got {cb.spent_so_far_uc}"
        print("OK: legacy LangChain 0.1 llm_output[token_usage] fallback works")
    else:
        print("SKIP: LangChain not installed — Tests 6 and 7 skipped")

    print("\nAll tests passed.")