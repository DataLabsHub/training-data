from celery import signature
from celery.execution import ExecutionEnvelope, propagate_envelope


def test_canvas_child_narrows_limits_and_survives_serialization():
    root = ExecutionEnvelope('root', tenant='t', deadline=200, retry_budget=4,
                             cancellation_scope='scope')
    child = propagate_envelope(root, label='group-0', deadline=150, retry_budget=2)
    assert child.execution_id != root.execution_id
    assert child.deadline == 150 and child.retry_budget == 2
    assert dict(child.extensions)['root_execution_id'] == 'root'
    sig = signature('demo.task', args=(1,)).with_execution(child)
    restored = signature(dict(sig)).execution_envelope
    assert restored == child
    narrowed = sig.constrain_execution(deadline=140, retry_budget=1)
    assert narrowed.execution_envelope.deadline == 140
