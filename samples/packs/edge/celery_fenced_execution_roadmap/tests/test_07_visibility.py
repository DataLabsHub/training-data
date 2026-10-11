import pytest

from kombu.common import VisibilityLeaseSet


def test_visibility_renew_freeze_settle_and_restore():
    now = [100.0]
    leases = VisibilityLeaseSet(clock=lambda: now[0], renewal_ratio=.5)
    leases.register('d1', execution_id='e', owner='w', generation=1, ttl=10)
    now[0] = 106
    assert leases.due()[0]['delivery_tag'] == 'd1'
    leases.freeze()
    assert leases.due() == ()
    with pytest.raises(RuntimeError):
        leases.renew('d1', owner='w', generation=1)
    snapshot = leases.snapshot()
    leases.thaw()
    assert leases.renew('d1', owner='w', generation=1)['renewals'] == 1
    assert leases.settle('d1') and not leases.settle('d1')
    other = VisibilityLeaseSet(clock=lambda: now[0])
    assert len(other.restore(snapshot)) == 1
