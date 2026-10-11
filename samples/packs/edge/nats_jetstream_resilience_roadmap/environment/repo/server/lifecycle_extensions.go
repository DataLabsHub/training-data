// Copyright 2025 The NATS Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import "time"

// RuntimeClock supplies wall-clock time to lifecycle-sensitive server paths.
// Embedders may provide a deterministic implementation. Implementations must
// be safe for concurrent use.
type RuntimeClock interface {
	Now() time.Time
}

// RuntimeClockFunc adapts a function to RuntimeClock.
type RuntimeClockFunc func() time.Time

// Now implements RuntimeClock.
func (f RuntimeClockFunc) Now() time.Time { return f() }

type systemRuntimeClock struct{}

func (systemRuntimeClock) Now() time.Time { return time.Now() }

// LifecycleEvent is an immutable snapshot of a server or JetStream lifecycle
// transition. Detail values are copied before delivery.
type LifecycleEvent struct {
	Sequence uint64            `json:"sequence"`
	Time     time.Time         `json:"time"`
	Kind     string            `json:"kind"`
	Server   string            `json:"server,omitempty"`
	Account  string            `json:"account,omitempty"`
	Stream   string            `json:"stream,omitempty"`
	Consumer string            `json:"consumer,omitempty"`
	Detail   map[string]string `json:"detail,omitempty"`
}

// LifecycleObserver receives lifecycle records. TryObserveLifecycle must not
// block. Returning false records a dropped observation without affecting the
// server operation that produced it.
type LifecycleObserver interface {
	TryObserveLifecycle(LifecycleEvent) bool
}

const (
	LifecycleServerCreated   = "server_created"
	LifecycleStreamCreated   = "stream_created"
	LifecycleStreamDeleted   = "stream_deleted"
	LifecycleConsumerCreated = "consumer_created"
	LifecycleConsumerDeleted = "consumer_deleted"
)

func (s *Server) runtimeNow() time.Time {
	if s == nil || s.runtimeClock == nil {
		return time.Now().UTC()
	}
	return s.runtimeClock.Now().UTC()
}

func (s *Server) emitLifecycle(kind, account, stream, consumer string, detail map[string]string) {
	if s == nil || s.lifecycleObserver == nil {
		return
	}
	var copied map[string]string
	if len(detail) > 0 {
		copied = make(map[string]string, len(detail))
		for k, v := range detail {
			copied[k] = v
		}
	}
	e := LifecycleEvent{
		Sequence: s.lifecycleSequence.Add(1),
		Time:     s.runtimeNow(),
		Kind:     kind,
		Server:   s.info.Name,
		Account:  account,
		Stream:   stream,
		Consumer: consumer,
		Detail:   copied,
	}
	if !s.lifecycleObserver.TryObserveLifecycle(e) {
		s.lifecycleDropped.Add(1)
	}
}

// LifecycleEventsDropped reports observer records rejected by the configured
// non-blocking sink.
func (s *Server) LifecycleEventsDropped() uint64 {
	if s == nil {
		return 0
	}
	return s.lifecycleDropped.Load()
}
