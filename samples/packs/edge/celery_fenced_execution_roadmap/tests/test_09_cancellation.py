from celery.execution import CancellationRegistry
from roadmap_helpers import clock


def test_cancellation_is_hierarchical_idempotent_and_inherited():
    scopes = CancellationRegistry(clock=clock())
    scopes.create('root')
    scopes.create('a', 'root')
    scopes.create('b', 'root')
    child, changed = scopes.cancel('a', 'child-only')
    assert changed and child.reason == 'child-only'
    assert not scopes.is_cancelled('root') and not scopes.is_cancelled('b')
    root, changed = scopes.cancel('root', 'shutdown')
    assert changed and scopes.is_cancelled('b')
    again, changed = scopes.cancel('root', 'later')
    assert not changed and again.reason == 'shutdown'
    late, _ = scopes.create('late', 'root')
    assert late.cancelled and late.reason == 'shutdown'
    assert CancellationRegistry.restore(scopes.snapshot()).is_cancelled('late')
