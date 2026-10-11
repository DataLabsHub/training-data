import pytest

from celery.execution import CursorError, ReplayCursor, replay_after
from roadmap_helpers import clock


def test_replay_cursor_is_monotonic_bounded_and_fail_closed():
    cursor = ReplayCursor(schedule_revision=4, sequence=1)
    decoded = ReplayCursor.decode(cursor.encode())
    assert decoded == cursor
    events = [{'sequence': 3, 'event_id': 'c'}, {'sequence': 2, 'event_id': 'b'}, {'sequence': 9, 'event_id': 'z'}]
    batch, advanced = replay_after(events, cursor, batch_limit=2)
    assert [item['sequence'] for item in batch] == [2, 3] and advanced.sequence == 3
    assert replay_after(events, advanced, batch_limit=1, deadline=999, clock=clock())[0] == ()
    with pytest.raises(CursorError):
        advanced.advance(sequence=2)
    with pytest.raises(CursorError):
        ReplayCursor.decode('not-a-cursor')
