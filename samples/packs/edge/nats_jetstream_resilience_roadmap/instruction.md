# JetStream resilience and lifecycle roadmap

This repository is a production NATS-compatible message server. Its JetStream path accepts protocol messages, sequences them into memory or file stores, replicates state through Raft, serves consumers and direct-get clients, and exposes account, monitoring, advisory, and tracing surfaces.

Implement the following roadmap as one integrated evolution. Preserve existing source compatibility where a method can reasonably accept both the legacy and new call shape. Do not replace the repository with another release: this fork contains local lifecycle contracts that must remain active.

## Shared rules

- All client-visible errors must use the normal API response/error path; malformed input must not panic or partially mutate state.
- Memory and file stores must agree on sequence, subject, TTL, purge, and restart semantics.
- Cluster followers must apply replicated decisions without independently creating divergent decisions.
- Timers, queues, callbacks, and subscriptions must be stopped or released when their owner stops, is deleted, or loses the relevant role.
- JSON additions must be backward compatible: omitted fields preserve prior behavior and unknown future metadata is retained where the existing API retains metadata.
- Maintain the `RuntimeClock` and `LifecycleObserver` options and all existing event behavior. Observer refusal must never block the operation that emitted the record.

## Dependency chains

1. T1 → T2 → T3 → T4 shares TTL parsing, store sequence state, timers, restart recovery, and subject lifecycle.
2. T5 → T6 → T7 → T8 shares consumer configuration, pull-request state, timers, advisories, and delivery eligibility.
3. T9 → T10 → T11 → T12 shares direct subscriptions, stream metadata, capability gates, placement, and leadership state.
4. T13 → T14 → T15 → T16 shares bounded queues, replicated envelopes, observability, deterministic time, and resource-release invariants.

## Targets

### T1 — Per-message TTL protocol contract

1. Recognize `Nats-TTL` only on streams that explicitly enable message TTLs.
2. Accept whole-second and duration spellings and normalize them to seconds.
3. Treat `never` as an explicit non-expiring override even when stream `MaxAge` is set.
4. Reject zero, negative, sub-second, malformed, and overflowing TTL values through the normal publish error response.
5. Preserve legacy publishes with no TTL header and do not leak invalid messages into storage or replication.

### T2 — Memory-store TTL lifecycle

1. Schedule per-message deadlines without changing public stream sequence numbers.
2. Expire due records while retaining later records and correct first/last state.
3. Remove timer entries on explicit delete, purge, compact, truncate, and stream stop.
4. Combine `MaxAge` and per-message TTL so the earliest applicable deadline wins, except for `never`.
5. Keep storage callbacks, per-subject counts, byte accounting, and idempotent repeated expiry correct.

### T3 — File-store TTL persistence and recovery

1. Persist enough TTL state to survive a clean restart without changing stored message encoding compatibility.
2. Rebuild missing or stale TTL-wheel state by scanning authoritative message headers and sequence bounds.
3. Expire overdue records after restart and schedule future records at their original deadlines.
4. Tolerate deleted, compacted, or truncated sequences during recovery without resurrecting them.
5. Keep block, stream, per-subject, tombstone, and byte accounting consistent after recovery expiry.

### T4 — Subject delete markers

1. Optionally create a marker when the last ordinary message for a subject is removed by limits or expiry.
2. Encode the removal reason and assign the configured marker TTL.
3. Never create a marker while another ordinary message for that subject remains.
4. Do not recursively replace an expired/deleted marker with another marker.
5. Preserve marker semantics across memory/file storage, purge/compact paths, restart, source/mirror processing, and replication.

### T5 — Consumer pause lifecycle

1. Accept a UTC pause deadline during consumer creation or update.
2. Keep pull and push consumers from delivering user data before the deadline.
3. Resume automatically at the deadline and make repeated resume operations idempotent.
4. Preserve pause state through consumer leadership changes and file-backed restart.
5. Emit pause/resume state and advisories without suppressing heartbeats or leaking timers.

### T6 — Priority-group configuration

1. Add stable JSON forms for `none`, `overflow`, and `pinned_client` policies.
2. Require at least one non-empty, syntactically valid group when a priority policy needs one.
3. Preserve group order and state in consumer info and clustered configuration snapshots.
4. Reject policy-specific pull fields on standard consumers and reject unknown groups.
5. Keep ordinary consumers and consumers that merely carry groups with `none` policy backward compatible.

### T7 — Pinned-client ownership

1. Assign an opaque pin ID to the first eligible request for a pinned group.
2. Return that ID on delivered messages and require it on subsequent owner requests.
3. Reject a mismatched ID without consuming or reordering a message.
4. Refresh ownership only on valid activity and release it after the configured TTL, deletion, stop, or role loss.
5. Allow a new owner after release and emit consistent pinned/unpinned advisory state.

### T8 — Overflow priority eligibility

1. Accept minimum pending and minimum ack-pending thresholds only for overflow consumers.
2. Deliver only when the configured/requested overflow threshold is met.
3. Return a normal timeout when it is not met, without reserving a message indefinitely.
4. Re-evaluate waiting requests as pending and ack-pending state changes.
5. Preserve fairness and release request resources on timeout, cancellation, consumer deletion, and leadership change.

### T9 — Bounded direct batch retrieval

1. Add a direct-get batch starting at an explicit sequence or next matching subject.
2. Return messages in stream order with subject, sequence, timestamp, and decreasing pending metadata.
3. Stop at the requested batch, byte limit, stream end, or server pending-output safety limit.
4. End every multi-response batch with one unambiguous status/EOB frame.
5. Reject incompatible selectors and require direct access to be enabled without changing single-get behavior.

### T10 — Multi-subject last-value retrieval

1. Resolve the last value for each requested subject filter in one direct request.
2. Deduplicate overlapping filters and return deterministic stream-sequence order.
3. Support an inclusive `up_to_seq` or `up_to_time` historical bound, but not both.
4. Return an explicit no-results frame and enforce the maximum result limit before excessive allocation.
5. Match memory and file stores, including deleted sequences, wildcards, pagination, and EOB metadata.

### T11 — Asset capability versioning

1. Store a required API level in stream and consumer metadata when a new capability is used.
2. Compute the maximum required level across all configured capabilities rather than last-writer wins.
3. Strip dynamic server version/level keys before persistence while retaining user metadata.
4. Add current server version/level only to response copies, never by mutating authoritative config.
5. Prevent leadership/placement on a server below the asset requirement while permitting legacy level-zero assets.

### T12 — Preferred leadership and catch-up safety

1. Accept an optional preferred server in placement and step-down requests.
2. Reject missing, non-member, ineligible, or below-capability candidates deterministically.
3. Do not report a successful transfer until the preferred peer is current enough to lead.
4. Preserve availability by leaving the current leader active when transfer validation or catch-up fails.
5. Cleanly cancel transfer/catch-up resources during superseding requests, deletion, shutdown, or topology change.

### T13 — Bounded stream-ingress queues

1. Bound internal stream ingestion by both queued message count and calculated byte size.
2. Reject the item that would exceed a bound without corrupting queue length or size accounting.
3. Return a client-visible 429 publish acknowledgement when a reply is available.
4. Release capacity precisely after pop, drain, recycle, stream deletion, and shutdown.
5. Preserve configured limits and safe defaults, including concurrent producers and an empty-to-nonempty notification transition.

### T14 — Replication envelope evolution

1. Extend replicated stream-message envelopes with forward-compatible flags, including source/mirror provenance.
2. Decode legacy flag-free envelopes with their previous meaning.
3. Preserve subject, reply, header, body, logical sequence, timestamp, and flags exactly across encode/decode.
4. Reject truncated, length-overflow, corrupt-compressed, and unknown mandatory forms without partial apply.
5. Apply replicated TTL/marker decisions idempotently and avoid follower-generated duplicate marker proposals.

### T15 — Distributed message tracing

1. Recognize explicit NATS trace destinations and sampled W3C `traceparent` headers case-insensitively.
2. Ignore unsampled or malformed external contexts unless an explicit valid NATS trace request exists.
3. Record ingress, subject mapping, service/stream import-export, route/gateway/leaf, JetStream, and egress transitions.
4. Preserve connection/server identity, timestamps, errors, and hop ordering while excluding authentication secrets.
5. Respect account destination/sampling policy, trace-only behavior, compression, and mixed-capability network links.

### T16 — Deterministic bounded expiry-work ledger (original)

1. Add an optional per-stream expiry work budget; zero preserves the historical unbounded pass.
2. Share the budget across `MaxAge` and per-message TTL work and order equal deadlines by sequence.
3. Report epoch, pass, processed count, remaining-due lower bound, storage kind, backlog, recovery, and cancellation state.
4. Continue backlog promptly while yielding between passes, and stop continuation on empty store, deletion, shutdown, or lost ownership.
5. Use `RuntimeClock` for deterministic pass decisions and send pass/backlog/recovery/cancellation records through `LifecycleObserver` without blocking storage.

## Completion standard

All targets must operate together for memory and file streams, remain deterministic under the injected clock, preserve both local extension contracts, compile with the existing Go toolchain, and pass package tests with the race detector. The verifier has independent target phases; a missing later API must not prevent earlier targets from receiving credit.
