package server

import (
	"reflect"
	"testing"
	"time"

	"github.com/nats-io/nats-server/v2/server/thw"
)

func TestRoadmapPhase16(t *testing.T) {
	now := time.Date(2026, 8, 24, 1, 2, 3, 0, time.UTC)
	wheel := thw.NewHashWheel()
	for seq, delta := range map[uint64]int64{4: -2, 3: -1, 2: -2, 1: 1} { if err := wheel.Add(seq, now.UnixNano()+delta); err != nil { t.Fatal(err) } }
	var order []uint64
	processed, remaining := wheel.ExpireTasksLimit(now.UnixNano(), 2, func(seq uint64, _ int64) { order = append(order, seq) })
	if processed != 2 || remaining != 1 || !reflect.DeepEqual(order, []uint64{2, 4}) { t.Fatalf("wheel pass = %d %d %v", processed, remaining, order) }
	ms, err := newMemStore(&StreamConfig{Name: "LEDGER", Storage: MemoryStorage, AllowMsgTTL: true, ExpiryWorkBudget: 2}); if err != nil { t.Fatal(err) }; defer ms.Stop()
	var updates []ExpiryWorkState
	ms.RegisterExpiryWorkUpdates(func(s ExpiryWorkState) { updates = append(updates, s) }, RuntimeClockFunc(func() time.Time { return now }))
	for seq := uint64(1); seq <= 5; seq++ { if err := ms.StoreRawMsg("a", nil, []byte("x"), seq, now.Add(-2*time.Second).UnixNano(), 1); err != nil { t.Fatal(err) } }
	ms.mu.Lock(); ms.cancelAgeChk(); ms.mu.Unlock()
	for want := uint64(3); want > 0; want-- { ms.expireMsgs(); ms.mu.Lock(); ms.cancelAgeChk(); ms.mu.Unlock(); if ms.ExpiryWorkState().Pass != 4-want { t.Fatal("pass counter did not advance") } }
	state := ms.ExpiryWorkState()
	if state.Processed != 1 || state.Remaining != 0 || state.Backlog || state.Epoch == 0 || len(updates) != 3 || ms.State().Msgs != 0 { t.Fatalf("ledger=%+v updates=%d store=%+v", state, len(updates), ms.State()) }
}
