from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Callable, Any

T = TypeVar("T")


class AffineViolation(RuntimeError):
    """Raised when a Budget is used after being consumed."""


class BudgetExhausted(RuntimeError):
    """Raised when a spend or split would exceed the cap."""


@dataclass
class Budget:
    """An affine budget capability.

    Once a method consumes `self` (spend, split, merge_with),
    subsequent uses raise AffineViolation. The intended usage is:

        budget = Budget(initial_uc=1000, max_uc=10_000)
        budget, after = budget.spend(100)
        # `before` is no longer usable; `budget` is the new one
    """

    initial_uc: int
    max_uc: int
    _consumed: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        if self.initial_uc > self.max_uc:
            raise ValueError(f"initial {self.initial_uc} exceeds max {self.max_uc}")
        if self.initial_uc < 0:
            raise ValueError("initial must be non-negative")

    def _check_alive(self) -> None:
        if self._consumed:
            raise AffineViolation(
                "Budget has been consumed; obtain the returned Budget "
                "from the previous spend/split call"
            )

    def micro_cents(self) -> int:
        with self._lock:
            self._check_alive()
            return self.initial_uc

    def spend(self, amount_uc: int) -> "Budget":
        """Spend `amount_uc` micro-cents. Consumes self; returns
        a fresh Budget with `initial - amount` remaining."""
        with self._lock:
            self._check_alive()
            if amount_uc < 0:
                raise ValueError("amount must be non-negative")
            if amount_uc > self.initial_uc:
                raise BudgetExhausted(
                    f"requested {amount_uc} uc, only {self.initial_uc} available"
                )
            self._consumed = True
            return Budget(initial_uc=self.initial_uc - amount_uc, max_uc=self.max_uc)

    def split(self, amount_uc: int) -> tuple["Budget", "Budget"]:
        """Split into (taken, kept). Consumes self."""
        with self._lock:
            self._check_alive()
            if amount_uc < 0 or amount_uc > self.initial_uc:
                raise BudgetExhausted(
                    f"split {amount_uc} out of {self.initial_uc} not possible"
                )
            self._consumed = True
            taken = Budget(initial_uc=amount_uc, max_uc=self.max_uc)
            kept = Budget(initial_uc=self.initial_uc - amount_uc, max_uc=self.max_uc)
            return taken, kept

    def merge_with(self, other: "Budget") -> "Budget":
        """Merge `other` into self. Consumes both."""
        with self._lock, other._lock:
            self._check_alive()
            other._check_alive()
            if self.max_uc != other.max_uc:
                raise ValueError("budgets must have matching max_uc")
            total = self.initial_uc + other.initial_uc
            if total > self.max_uc:
                raise BudgetExhausted(
                    f"merge would exceed max {self.max_uc}: {total}"
                )
            self._consumed = True
            other._consumed = True
            return Budget(initial_uc=total, max_uc=self.max_uc)


class BudgetPool:
    """Multi-tenant pool with closure-based reservation API.

    The `with_reservation` method REQUIRES the closure to call
    `receipt.confirm(...)` or `receipt.forfeit(...)` before
    returning. Failure to do so raises AffineViolation at exit.
    """

    def __init__(self, available_uc: int):
        self.available_uc = available_uc
        self.outstanding_uc = 0
        self._lock = threading.Lock()

    def with_reservation(
            self,
            amount_uc: int,
            callback: Callable[["ReservationReceipt"], "ResolvedReceipt[T]"],
    ) -> T:
        receipt = self._reserve_internal(amount_uc)
        try:
            resolved = callback(receipt)
        except Exception:
            # Closure raised before resolving — forfeit
            self._forfeit_internal(receipt.reserved_uc)
            raise
        if not isinstance(resolved, ResolvedReceipt):
            # Closure forgot to confirm/forfeit
            self._forfeit_internal(receipt.reserved_uc)
            raise AffineViolation(
                "callback did not return a ResolvedReceipt; receipt was "
                "auto-forfeited. Call receipt.confirm(...) or "
                "receipt.forfeit(...) before returning."
            )
        return resolved.inner

    def _reserve_internal(self, amount_uc: int) -> "ReservationReceipt":
        with self._lock:
            if amount_uc > self.available_uc:
                raise BudgetExhausted(
                    f"pool has only {self.available_uc} uc, requested {amount_uc}"
                )
            self.available_uc -= amount_uc
            self.outstanding_uc += amount_uc
        return ReservationReceipt(self, amount_uc)

    def _confirm_internal(self, reserved_uc: int, actual_uc: int) -> None:
        with self._lock:
            assert actual_uc <= reserved_uc
            self.outstanding_uc -= reserved_uc
            self.available_uc += reserved_uc - actual_uc

    def _forfeit_internal(self, reserved_uc: int) -> None:
        with self._lock:
            self.outstanding_uc -= reserved_uc


class ReservationReceipt:
    def __init__(self, pool: BudgetPool, reserved_uc: int):
        self.pool = pool
        self.reserved_uc = reserved_uc
        self._resolved = False

    def confirm(self, actual_uc: int, value: T) -> "ResolvedReceipt[T]":
        if self._resolved:
            raise AffineViolation("receipt already resolved")
        if actual_uc > self.reserved_uc:
            raise BudgetExhausted(
                f"actual {actual_uc} exceeds reserved {self.reserved_uc}"
            )
        self.pool._confirm_internal(self.reserved_uc, actual_uc)
        self._resolved = True
        return ResolvedReceipt(value, _private=_PRIVATE_TOKEN)

    def forfeit(self, value: T) -> "ResolvedReceipt[T]":
        if self._resolved:
            raise AffineViolation("receipt already resolved")
        self.pool._forfeit_internal(self.reserved_uc)
        self._resolved = True
        return ResolvedReceipt(value, _private=_PRIVATE_TOKEN)


_PRIVATE_TOKEN = object()


@dataclass
class ResolvedReceipt(Generic[T]):
    """Witness that a receipt was resolved. Can only be
    constructed by ReservationReceipt.confirm/forfeit."""

    inner: T
    _private: Any = None

    def __post_init__(self):
        if self._private is not _PRIVATE_TOKEN:
            raise AffineViolation(
                "ResolvedReceipt cannot be constructed directly; use "
                "ReservationReceipt.confirm() or forfeit()"
            )


class LangChainBudgetCallback:
    """LangChain BaseCallbackHandler that bounds total cost using
    a Budget.

    Usage:
        budget = Budget(initial_uc=10_000, max_uc=100_000)
        cb = LangChainBudgetCallback(budget, rate_per_input_token_uc=15,
                                     rate_per_output_token_uc=60)
        agent.invoke({...}, config={"callbacks": [cb]})
        # If the agent's running spend exceeds the budget,
        # cb raises BudgetExhausted which aborts the chain.
    """

    def __init__(
            self,
            budget: Budget,
            rate_per_input_token_uc: int = 15,
            rate_per_output_token_uc: int = 60,
    ):
        self._budget = budget
        self.rate_in = rate_per_input_token_uc
        self.rate_out = rate_per_output_token_uc
        self._spent_so_far = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Pre-flight estimate
        est = sum(len(p) for p in prompts) * self.rate_in
        if self._spent_so_far + est > self._budget.micro_cents():
            raise BudgetExhausted(
                f"pre-flight estimate {est} would exceed remaining "
                f"{self._budget.micro_cents() - self._spent_so_far}"
            )

    def on_llm_end(self, response, **kwargs):
        usage = getattr(response, "llm_output", {}).get("token_usage", {})
        if not usage:
            return
        cost = (
                usage.get("prompt_tokens", 0) * self.rate_in
                + usage.get("completion_tokens", 0) * self.rate_out
        )
        self._spent_so_far += cost
        if self._spent_so_far > self._budget.micro_cents():
            raise BudgetExhausted(
                f"running spend {self._spent_so_far} exceeded budget"
            )


if __name__ == "__main__":
    # Test 1: basic spend
    b = Budget(initial_uc=1000, max_uc=10_000)
    b2 = b.spend(100)
    assert b2.micro_cents() == 900

    try:
        b.spend(50)
    except AffineViolation:
        print("OK: double-spend detected")
    else:
        raise AssertionError("expected AffineViolation")

    b = Budget(initial_uc=1000, max_uc=10_000)
    taken, kept = b.split(300)
    assert taken.micro_cents() + kept.micro_cents() == 1000

    # Test 4: pool with_reservation
    pool = BudgetPool(available_uc=10_000)
    result = pool.with_reservation(
        500, lambda r: r.confirm(423, "agent output")
    )
    assert result == "agent output"

    try:
        pool.with_reservation(500, lambda r: "forgot to resolve")
    except AffineViolation:
        print("OK: forgot-to-resolve detected at runtime")
    else:
        raise AssertionError("expected AffineViolation")

    print("All tests passed.")