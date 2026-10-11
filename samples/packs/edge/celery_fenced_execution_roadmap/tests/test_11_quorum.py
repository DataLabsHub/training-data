import pytest

from celery.execution import CheckpointDecision, QuorumCheckpoint


def test_quorum_opens_or_becomes_impossible_exactly_once():
    checkpoint = QuorumCheckpoint('q', ['a', 'b', 'c'], quorum=2)
    assert checkpoint.record('a', True)[0] == CheckpointDecision.PENDING
    assert checkpoint.record('a', True)[1] is False
    assert checkpoint.record('b', True)[0] == CheckpointDecision.OPEN
    restored = QuorumCheckpoint.restore(checkpoint.encode())
    assert restored.decision == CheckpointDecision.OPEN and restored.successes == 2
    impossible = QuorumCheckpoint('bad', ['a', 'b', 'c'], quorum=2)
    impossible.record('a', False)
    assert impossible.record('b', False)[0] == CheckpointDecision.IMPOSSIBLE
    with pytest.raises(KeyError):
        impossible.record('unknown', True)
