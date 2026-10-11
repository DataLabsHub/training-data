import pytest

from celery.execution import CompletionState, CompletionTokenStore, LeaseConflict, QuorumCheckpoint
from roadmap_helpers import clock


def test_completion_token_is_fenced_resumable_and_at_most_once():
    checkpoint = QuorumCheckpoint('q', ['a'], quorum=1, generation=3)
    checkpoint.record('a', True)
    tokens = CompletionTokenStore(clock=clock())
    token, created = tokens.acquire(checkpoint, 'w', generation=3)
    again, duplicate = tokens.acquire(checkpoint, 'w', generation=3)
    assert created and not duplicate and token == again and tokens.pending() == (token,)
    published, changed = tokens.mark_published('q', 'w', 3)
    assert changed and published.state == CompletionState.PUBLISHED
    assert tokens.mark_published('q', 'w', 3)[1] is False
    with pytest.raises(LeaseConflict):
        tokens.mark_published('q', 'old', 3)
