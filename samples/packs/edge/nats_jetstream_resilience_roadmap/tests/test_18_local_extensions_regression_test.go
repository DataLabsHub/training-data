package server

import (
	"testing"
	"time"
)

type roadmapObserver struct { events []LifecycleEvent; accept bool }
func (o *roadmapObserver) TryObserveLifecycle(e LifecycleEvent) bool { o.events = append(o.events, e); return o.accept }

func TestRoadmapPhase18(t *testing.T) {
	fixed := time.Date(2031, 2, 3, 4, 5, 6, 0, time.FixedZone("test", 8*3600))
	observer := &roadmapObserver{accept: true}
	s, _ := roadmapRunServer(t, func(opts *Options) { opts.RuntimeClock = RuntimeClockFunc(func() time.Time { return fixed }); opts.LifecycleObserver = observer })
	if len(observer.events) == 0 || observer.events[0].Kind != LifecycleServerCreated || !observer.events[0].Time.Equal(fixed.UTC()) || observer.events[0].Sequence != 1 { t.Fatalf("events = %+v", observer.events) }
	if !s.start.Equal(fixed.UTC()) { t.Fatalf("server start = %v", s.start) }
	reject := &roadmapObserver{}
	s2, _ := roadmapRunServer(t, func(opts *Options) { opts.RuntimeClock = RuntimeClockFunc(func() time.Time { return fixed }); opts.LifecycleObserver = reject })
	if s2.LifecycleEventsDropped() != 1 || len(reject.events) != 1 { t.Fatalf("drop accounting = %d events=%d", s2.LifecycleEventsDropped(), len(reject.events)) }
}
