from celery import Celery
from celery.local import ExecutionObserver, FrozenClock


def test_local_observer_extension_is_bounded_and_non_failing():
    def broken(event):
        raise RuntimeError(event.kind)

    app = Celery('observer-regression', broker='memory://')
    observer = ExecutionObserver(broken, max_events=2)
    app.set_clock_source(FrozenClock(wall=100))
    previous = app.set_execution_observer(observer)
    assert previous is not observer
    assert app.execution_observer is observer
    app.send_task('one', task_id='1')
    app.send_task('two', task_id='2')
    app.send_task('three', task_id='3')
    events = observer.snapshot()
    assert [event.task_id for event in events] == ['2', '3']
    assert all(event.kind == 'task.published' for event in events)
    assert len(observer.errors()) == 2
