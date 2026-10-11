package server

import (
	"testing"
	"time"
)

func TestRoadmapPhase02(t *testing.T) {
	ms, err := newMemStore(&StreamConfig{Name: "MEMTTL", Storage: MemoryStorage, AllowMsgTTL: true})
	if err != nil { t.Fatal(err) }
	defer ms.Stop()
	now := time.Now()
	if err := ms.StoreRawMsg("a", nil, []byte("expired"), 1, now.Add(-3*time.Second).UnixNano(), 1); err != nil { t.Fatal(err) }
	if err := ms.StoreRawMsg("a", nil, []byte("live"), 2, now.UnixNano(), 60); err != nil { t.Fatal(err) }
	ms.expireMsgs()
	if _, err := ms.LoadMsg(1, nil); err == nil { t.Fatal("expired message remained") }
	if msg, err := ms.LoadMsg(2, nil); err != nil || string(msg.msg) != "live" { t.Fatalf("live message: %v, %v", msg, err) }
	if state := ms.State(); state.Msgs != 1 || state.FirstSeq != 2 { t.Fatalf("state = %+v", state) }
}
