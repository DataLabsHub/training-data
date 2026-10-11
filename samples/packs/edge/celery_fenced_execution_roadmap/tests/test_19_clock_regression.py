from celery import Celery
from celery.local import FrozenClock


def test_local_clock_extension_and_publish_header_remain_compatible():
    app = Celery('clock-regression', broker='memory://')
    frozen = FrozenClock(wall=1234, monotonic=10)
    previous = app.set_clock_source(frozen)
    assert previous is not frozen
    assert app.clock_source is frozen
    result = app.send_task('missing.task', task_id='known-id')
    assert result.id == 'known-id'
    frozen.advance(5)
    assert frozen.snapshot() == {'wall': 1239.0, 'monotonic': 15.0}
