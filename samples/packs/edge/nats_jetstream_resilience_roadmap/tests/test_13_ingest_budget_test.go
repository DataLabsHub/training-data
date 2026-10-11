package server

import (
	"errors"
	"testing"
)

func TestRoadmapPhase13(t *testing.T) {
	s := &Server{}
	q := newIPQueue[int](s, "bounded", ipqSizeCalculation(func(v int) uint64 { return uint64(v) }), ipqLimitByLen[int](2), ipqLimitBySize[int](5))
	if n, err := q.push(2); n != 1 || err != nil { t.Fatalf("push1 = %d %v", n, err) }
	if n, err := q.push(3); n != 2 || err != nil { t.Fatalf("push2 = %d %v", n, err) }
	if n, err := q.push(1); n != 2 || !errors.Is(err, errIPQLenLimitReached) { t.Fatalf("len limit = %d %v", n, err) }
	if got, ok := q.popOne(); !ok || got != 2 { t.Fatalf("pop = %d %v", got, ok) }
	if n, err := q.push(4); n != 1 || !errors.Is(err, errIPQSizeLimitReached) { t.Fatalf("size limit = %d %v", n, err) }
	if n, err := q.push(2); n != 2 || err != nil { t.Fatalf("capacity was not released = %d %v", n, err) }
}
