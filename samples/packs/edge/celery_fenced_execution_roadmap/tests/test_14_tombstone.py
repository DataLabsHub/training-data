import pytest

from celery.execution import ResultConflict, VersionedResultStore
from roadmap_helpers import clock


def test_tombstone_fences_stale_writes_and_audits_repair():
    c = clock()
    store = VersionedResultStore(clock=c, tombstone_ttl=10)
    store.compare_and_set('e', 'SUCCESS', 1, expected_version=0, generation=2)
    tomb, changed = store.forget('e', generation=2)
    assert changed and tomb.state == 'TOMBSTONE' and tomb.version == 2
    assert store.forget('e', generation=2)[1] is False
    with pytest.raises(ResultConflict):
        store.compare_and_set('e', 'SUCCESS', 2, generation=1)
    repaired, _ = store.compare_and_set('e', 'SUCCESS', 2, expected_version=2,
                                        generation=3, repair=True, audit_reason='operator')
    assert repaired.value == 2 and repaired.version == 3 and dict(repaired.audit[-1])['reason'] == 'operator'
