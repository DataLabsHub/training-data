from celery.execution import EventJournal


def test_journal_deduplicates_reorders_gaps_and_restores():
    journal = EventJournal(reorder_window=2)
    assert journal.append('e', 'two', 2, 'STARTED', timestamp=2)[0] == ()
    emitted, changed = journal.append('e', 'one', 1, 'RECEIVED', timestamp=1)
    assert changed and [event.sequence for event in emitted] == [1, 2]
    assert journal.append('e', 'one', 1, 'RECEIVED', timestamp=1)[1] is False
    journal.append('e', 'six', 6, 'SUCCESS', timestamp=6)
    assert journal.gaps('e')[0].as_dict() == {'execution_id': 'e', 'start': 3, 'end': 5}
    restored = EventJournal.restore(journal.encode())
    assert restored.cursor('e') == 6 and restored.terminal('e') == 'SUCCESS'
