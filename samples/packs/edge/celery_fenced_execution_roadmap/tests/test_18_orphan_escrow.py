from celery.execution import ExecutionEnvelope, OrphanDecision, OrphanEscrow
from roadmap_helpers import clock, orphan_evidence


def test_adaptive_orphan_escrow_requires_convergence_and_restores_fairness():
    c = clock()
    escrow = OrphanEscrow(clock=c, max_per_tenant=2, base_allowance=2, minimum_allowance=1)
    env_a = ExecutionEnvelope('a', tenant='a', generation=1, retry_budget=2, deadline=1100)
    env_b = ExecutionEnvelope('b', tenant='b', generation=4, retry_budget=1, deadline=1100)
    weak = dict(orphan_evidence(), pool_missing=False)
    assert escrow.submit(env_a, **weak)[0] == OrphanDecision.NOT_ORPHAN
    assert escrow.submit(env_a, **orphan_evidence()) == (OrphanDecision.ESCROW, True)
    assert escrow.submit(env_b, **orphan_evidence()) == (OrphanDecision.ESCROW, True)
    restored = OrphanEscrow.restore(escrow.encode(), clock=c)
    before = escrow.release(2)
    after = restored.release(2)
    assert before == after and {item.tenant for item in before} == {'a', 'b'}
    assert all(item.retry_budget >= 0 and item.generation > 1 for item in before)
    assert restored.quarantine_late_result('b', 4, 5, value='late')
