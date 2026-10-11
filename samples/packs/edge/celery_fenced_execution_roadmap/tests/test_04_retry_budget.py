import pytest

from celery.execution import ExecutionExpired, RetryBudget, RetryBudgetExhausted


def test_retry_budget_is_deterministic_bounded_and_deadline_aware():
    budget = RetryBudget(remaining=2, base_delay=2, maximum_delay=3)
    delay1, next1 = budget.consume('e', now=10, deadline=100)
    delay2, next2 = budget.consume('e', now=10, deadline=100)
    assert delay1 == delay2 and 0 <= delay1 <= 3
    assert next1 == next2 and next1.remaining == 1
    _, empty = next1.consume('e', now=10, deadline=100)
    with pytest.raises(RetryBudgetExhausted):
        empty.consume('e')
    with pytest.raises(ExecutionExpired):
        budget.consume('e', now=99.9, deadline=100)
