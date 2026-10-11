import pytest

from celery.execution import EnvelopeValidationError, resolve_execution_id


def test_identity_is_canonical_domain_separated_and_compatible():
    a = resolve_execution_id('t', (1, {'b': 2, 'a': 1}), {'z': [3, 4]}, tenant='x')
    b = resolve_execution_id('t', [1, {'a': 1, 'b': 2}], {'z': (3, 4)}, tenant='x')
    assert a == b and a.startswith('exec-')
    assert a != resolve_execution_id('t', [1, {'a': 1, 'b': 2}], {'z': (3, 4)}, tenant='y')
    assert a != resolve_execution_id('t', [1, {'a': 1, 'b': 2}], {'z': (3, 4)}, tenant='x', idempotency_key='k')
    assert resolve_execution_id('t', task_id='legacy-id') == 'legacy-id'
    with pytest.raises(EnvelopeValidationError):
        resolve_execution_id('t', [float('nan')])
