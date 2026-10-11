from celery.local import FrozenClock


def clock(value=1000.0):
    return FrozenClock(wall=value, monotonic=value)


def orphan_evidence():
    return dict(
        claim_missing=True,
        lease_missing=True,
        journal_nonterminal=True,
        pool_missing=True,
    )
