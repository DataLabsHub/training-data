import pytest

from celery.execution import ExecutionEnvelope, ExecutionExpired, absolute_deadline, ensure_live
from roadmap_helpers import clock


def test_deadline_normalizes_propagates_and_expires():
    c = clock()
    assert absolute_deadline(clock=c, countdown=20, time_limit=10, parent=1030) == 1010
    parent = ExecutionEnvelope('p', deadline=1020)
    child = parent.child('c', deadline=1015)
    assert child.deadline == 1015
    with pytest.raises(ExecutionExpired):
        absolute_deadline(clock=c, parent=999)
    c.advance(16)
    with pytest.raises(ExecutionExpired):
        ensure_live(child, clock=c)
