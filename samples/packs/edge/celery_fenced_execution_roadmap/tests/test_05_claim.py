import pytest

from celery.execution import ClaimConflict, ClaimStore
from roadmap_helpers import clock


def test_claim_idempotency_fencing_and_serialization():
    c = clock()
    store = ClaimStore(clock=c)
    first, created = store.claim('e', 'w1', generation=1, ttl=10)
    again, created_again = store.claim('e', 'w1', generation=1, ttl=10)
    assert created and not created_again and first == again
    with pytest.raises(ClaimConflict):
        store.claim('e', 'w2', generation=2, ttl=10)
    c.advance(11)
    with pytest.raises(ClaimConflict):
        store.claim('e', 'w2', generation=1, ttl=10)
    newer, _ = store.claim('e', 'w2', generation=2, ttl=10)
    restored = ClaimStore.decode(store.encode(), clock=c)
    assert restored.get('e') == newer
