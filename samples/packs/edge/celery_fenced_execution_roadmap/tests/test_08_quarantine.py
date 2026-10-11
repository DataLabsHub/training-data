from celery.execution import ClaimStore, DeliveryDecision, DeliveryFence, ExecutionEnvelope, VersionedResultStore
from roadmap_helpers import clock


def test_delivery_classification_uses_claim_terminal_and_deadline_evidence():
    c = clock()
    claims = ClaimStore(clock=c)
    claims.claim('e', 'w', generation=2, ttl=20)
    results = VersionedResultStore(clock=c)
    fence = DeliveryFence(claims, results, clock=c)
    assert fence.classify(ExecutionEnvelope('e', generation=1)) == DeliveryDecision.STALE
    assert fence.classify(ExecutionEnvelope('e', generation=2)) == DeliveryDecision.ACCEPT
    assert fence.classify(ExecutionEnvelope('e', generation=3)) == DeliveryDecision.FUTURE
    results.compare_and_set('e', 'SUCCESS', 42, expected_version=0, generation=2)
    assert fence.classify(ExecutionEnvelope('e', generation=2)) == DeliveryDecision.DUPLICATE
    assert fence.classify(ExecutionEnvelope('x', deadline=999)) == DeliveryDecision.EXPIRED
