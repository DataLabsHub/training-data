import pytest

from celery.execution import ResultConflict, VersionedResultStore
from roadmap_helpers import clock


def test_result_compare_and_set_versions_and_terminal_fence():
    store = VersionedResultStore(clock=clock())
    pending, _ = store.compare_and_set('e', 'STARTED', expected_version=0, generation=1)
    assert pending.version == 1
    done, _ = store.compare_and_set('e', 'SUCCESS', 42, expected_state='STARTED', expected_version=1, generation=1)
    assert done.version == 2 and done.value == 42
    with pytest.raises(ResultConflict) as conflict:
        store.compare_and_set('e', 'FAILURE', expected_version=2, generation=1)
    assert conflict.value.actual_state == 'SUCCESS' and conflict.value.actual_version == 2
    same, changed = store.compare_and_set('e', 'SUCCESS', 42, expected_version=2, generation=1)
    assert same == done and not changed
