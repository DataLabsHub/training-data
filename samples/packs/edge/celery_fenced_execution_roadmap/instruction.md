# Durable fenced execution roadmap

This repository is an integrated Celery product workspace.  The root contains
Celery, while `components/kombu`, `components/billiard`, and `components/vine`
are the broker, process-pool, and promise runtimes used by the root project.
They are developed and tested as one offline product.  Implement the roadmap
below in the existing publish, Canvas, transport, worker, pool, backend,
events, and scheduler paths.  A parallel package which is not used by those
paths is not a completed implementation.

The roadmap introduces one immutable execution contract shared by all four
components.  `execution_id`, `generation`, deadline, cancellation scope,
result version, event sequence, and durable lifecycle decisions must agree
across process and reconnect boundaries.  All encodings and scheduling
choices must be deterministic.

## Target 1 — canonical execution envelope

Provide an immutable, versioned envelope containing execution id, attempt,
generation, tenant, absolute deadline, retry budget, cancellation scope,
trace data, and extension data.  Reject unsupported schema versions and
negative counters.  Map order must not affect canonical bytes.  Bytes, text,
and mapping inputs must decode equally; unknown fields must round-trip as
extensions, and returned mappings must not expose mutable internal state.

## Target 2 — deterministic execution identity

Derive an id from task name, arguments, keyword arguments, tenant, and an
optional idempotency key.  Equal mappings and tuple/list argument spellings
must produce equal ids.  Tenant and idempotency key are identity domains.
Reject non-finite values.  An explicit legacy task id always wins.

## Target 3 — absolute deadline propagation

Normalize countdown, expires, time limit, and inherited limits to the
earliest UTC epoch deadline.  Propagate the earliest limit through chains,
groups, callbacks, and message headers.  Refuse already-expired publication
and prevent execution when expiry occurs before the task starts.  Every
comparison must use the application clock hook described under compatibility.

## Target 4 — retry budget and deterministic backoff

Consume one shared retry token for every retry request, reject exhausted
budgets, and provide capped exponential delay with execution-id-derived,
repeatable jitter.  Never schedule after the absolute deadline.  Tasks with no
execution envelope retain existing `max_retries` behavior.

## Target 5 — idempotent backend claim

Acquire an execution claim with owner, generation, and expiry.  Repeating the
same claim is idempotent.  A live claim cannot be stolen; after expiry only a
higher generation may acquire it.  Claims must have deterministic snapshot
and restoration support and be accessible to the worker/backend path.

## Target 6 — delivery lease with generation fencing

Open a delivery-tag lease when work is accepted.  Duplicate or lower
generations cannot supersede an active lease.  Only the active owner and
generation may renew it.  Closing is idempotent and releases its execution
index.  Expired leases cannot be revived by a late renewal.

## Target 7 — broker visibility renewal

Kombu QoS must track delivery tags, renewal deadlines, owners, and
generations.  Heartbeat renewal extends visibility; ack and reject remove
state exactly once.  Connection loss freezes renewal ordering.  Restoration
must ignore expired, settled, or older-generation records and broker
reconnects must advance their connection epoch.

## Target 8 — stale-delivery quarantine

Before user code runs, compare message generation with claim, cancellation,
deadline, and terminal-result evidence.  Older work and terminal redeliveries
must not execute; a future generation without a matching claim is rejected.
The delivery is settled and a structured quarantine decision is observable.

## Target 9 — hierarchical cancellation scopes

Support stable parent/child scopes.  Cancelling a parent cancels all current
descendants; cancelling a child leaves its parent and siblings live.  The
first reason and timestamp win.  Repetition is idempotent, and a child created
after its parent was cancelled starts cancelled.  Snapshot restoration must
reject cyclic or missing-parent state.

## Target 10 — Canvas envelope propagation

Signatures, chains, groups, links, errbacks, clones, freeze, and serialization
must preserve execution metadata and causality.  Group members have
deterministic child ids and retain their root id.  Replacement and child
construction may narrow, but never widen, deadlines or retry budgets.

## Target 11 — durable quorum checkpoint

Register fixed group membership and a required success quorum.  Each child
terminal result counts once.  Open exactly when quorum is satisfied; decide
impossible once the remaining children cannot make quorum.  Unknown members
and conflicting terminal reports are errors.  Snapshot/restore preserves the
terminal decision and counts.

## Target 12 — idempotent chord completion

Use a generation-fenced durable completion token.  An open checkpoint chooses
the body path; an impossible checkpoint chooses the errback path.  Publish
each at most once despite duplicate notifications.  Persist the decision
before publication so a crash can resume the same pending token.  Reject a
stale owner, generation, or changed path.

## Target 13 — compare-and-set result transitions

Result storage accepts a state transition only when the expected version and
state match, increments a monotonic version, and reports conflicts with actual
state/version.  Terminal states cannot transition normally.  Legacy
`store_result` calls without CAS fields continue to work.

## Target 14 — tombstones and stale-write isolation

Forget creates a versioned tombstone.  It hides prior data and fences older
worker generations.  Repeated forget is idempotent and tombstones have a
deterministic retention time.  A deliberate higher-generation repair may
resurrect data only with an audit record.

## Target 15 — sequenced lifecycle event journal

Assign monotonic per-execution sequence numbers and deduplicate event ids.
Buffer bounded out-of-order arrivals, emit contiguous events, and record a
deterministic gap after the reorder window is exceeded.  Journal snapshots
preserve emitted events, buffers, gaps, cursors, and terminal state.  Event
dispatch and receive paths must use these semantics.

## Target 16 — replay cursor and scheduler catch-up

Provide a versioned cursor containing schedule revision and last sequence.
Replay only later events in stable order, obey a positive batch limit and an
absolute deadline, and advance cursors monotonically and idempotently.  Beat
catch-up must persist the same state.  Corrupt, backward, or future-version
cursors fail closed.

## Target 17 — worker drain snapshot and restoration

Drain mode rejects new envelopes while allowing accepted work to settle.
Produce canonical snapshots of active leases, reservations, cancellation
scopes, admission state, and process-pool jobs.  Restoration discards expired,
cancelled, or stale-generation work and must not duplicate a job across
repeated restore.  Kombu connection and Billiard pool/process state must
participate in the handoff.

## Target 18 — adaptive orphan escrow recovery

This product-specific policy is required in addition to ordinary retry.
Classify an execution as orphaned only when claim, lease, nonterminal journal,
and process-pool evidence all agree.  Put recoverable work in bounded
per-tenant escrow.  Release with deterministic deficit round robin.  Reduce a
tenant's allowance after repeated orphan generations and recover it after
successful completions.  Never release after deadline, cancellation, or
budget exhaustion.  Increment generation and consume budget on release;
quarantine late results from prior generations.  Snapshot restoration must
preserve queues, adaptive allowances, and fairness credits so it produces the
same future release order.

## Dependencies and completion criteria

The principal dependency chains are T1→T2→T3→T4, T5→T6→T7→T8,
T9→T10→T11→T12, T13→T14→T15→T16, and T3/T6/T11→T17→T18.  Later targets may
use earlier contracts, but each target's failure and idempotency behavior is
part of its public behavior.  The complete system must operate through at
least application dispatch, Canvas, Kombu message/QoS/connection, worker
request/state/control, Billiard pool lifecycle, backend/result, events, and
beat/scheduler paths.

Preserve existing public APIs unless an additive parameter is necessary.
Existing Celery tests and the two local compatibility contracts must continue
to pass:

- `celery.local.ClockSource` / `FrozenClock`, configured per application, is
  the sole clock for execution decisions and existing dispatch timestamps.
- `celery.local.ExecutionObserver`, configured per application, remains a
  bounded non-failing audit sink used by publish and request paths.  Callback
  exceptions are recorded but never break task delivery.

Do not add network requirements, real credentials, nondeterministic sleeps,
or machine-specific paths.  The integrated repository must compile and test
offline using the checked-in component sources.  Thread-shared reference
stores must remain race-safe, all snapshots must be plain serializable data,
and no failure may be hidden by silently falling back to an unfenced path.
