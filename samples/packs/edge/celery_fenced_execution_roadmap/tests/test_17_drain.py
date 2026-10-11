import pytest

from celery.execution import DrainController, DrainError
from roadmap_helpers import clock


def test_drain_snapshot_filters_and_restore_is_idempotent():
    c = clock()
    drain = DrainController('w', clock=c)
    assert drain.begin() and not drain.begin()
    with pytest.raises(DrainError):
        drain.require_accepting()
    snapshot = drain.snapshot(
        leases=[{'delivery_tag': 'live', 'expires_at': 1100, 'generation': 2}, {'delivery_tag': 'old', 'expires_at': 999, 'generation': 2}],
        reservations=[{'execution_id': 'e', 'generation': 2, 'deadline': 1100}],
        pool_jobs=[{'execution_id': 'e', 'job_id': 'j'}],
    )
    assert snapshot.encode() == drain.snapshot(leases=reversed(snapshot.leases), reservations=snapshot.reservations, pool_jobs=snapshot.pool_jobs).encode()
    first = drain.restore(snapshot.encode(), minimum_generation=2)
    second = drain.restore(snapshot.encode(), minimum_generation=2)
    assert len(first.leases) == 1 and len(first.pool_jobs) == 1 and second.pool_jobs == ()
