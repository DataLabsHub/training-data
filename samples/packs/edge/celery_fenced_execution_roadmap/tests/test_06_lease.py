import pytest

from celery.execution import LeaseConflict, LeaseRegistry
from roadmap_helpers import clock


def test_delivery_lease_owner_generation_and_release():
    c = clock()
    leases = LeaseRegistry(clock=c)
    first, created = leases.open('e', 'tag', 'w', 2, 10)
    again, duplicate = leases.open('e', 'tag', 'w', 2, 10)
    assert created and not duplicate and first == again
    with pytest.raises(LeaseConflict):
        leases.open('e', 'other', 'old', 1, 10)
    c.advance(2)
    assert leases.renew('tag', 'w', 2, 20).expires_at == 1022
    with pytest.raises(LeaseConflict):
        leases.renew('tag', 'old', 2, 20)
    assert leases.close('tag', 'w', 2) is True
    assert leases.close('tag', 'w', 2) is False
