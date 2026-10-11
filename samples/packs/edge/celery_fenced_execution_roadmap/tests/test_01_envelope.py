import pytest

from celery.execution import ExecutionEnvelope, EnvelopeValidationError


def test_canonical_envelope_roundtrip_and_validation():
    value = {'execution_id': 'e1', 'tenant': 'acme', 'attempt': 2,
             'generation': 3, 'trace': {'b': 2, 'a': 1}, 'future': {'x': 1}}
    left = ExecutionEnvelope.from_value(value)
    right = ExecutionEnvelope.from_value(left.encode())
    assert left == right
    assert left.encode() == right.encode()
    assert dict(right.extensions)['future'] == {'x': 1}
    snapshot = right.as_dict()
    snapshot['execution_id'] = 'mutated'
    assert right.execution_id == 'e1'
    with pytest.raises(EnvelopeValidationError):
        ExecutionEnvelope('bad', schema=99)
    with pytest.raises(EnvelopeValidationError):
        ExecutionEnvelope('bad', attempt=-1)
