package server

import (
	"testing"
	"time"
)

func TestRoadmapPhase04(t *testing.T) {
	ms, err := newMemStore(&StreamConfig{Name: "MARKERS", Storage: MemoryStorage, MaxAge: time.Second, AllowMsgTTL: true, SubjectDeleteMarkerTTL: time.Second})
	if err != nil { t.Fatal(err) }
	defer ms.Stop()
	markers := make(chan *inMsg, 2)
	ms.RegisterSubjectDeleteMarkerUpdates(func(msg *inMsg) { markers <- msg })
	now := time.Now().Add(-2 * time.Second).UnixNano()
	for seq := uint64(1); seq <= 2; seq++ { if err := ms.StoreRawMsg("orders.eu", nil, []byte("x"), seq, now, 0); err != nil { t.Fatal(err) } }
	ms.mu.Lock(); ms.cancelAgeChk(); ms.mu.Unlock()
	ms.expireMsgs()
	select {
	case marker := <-markers:
		if marker.subj != "orders.eu" || string(getHeader(JSMarkerReason, marker.hdr)) == "" { t.Fatalf("bad marker: %+v", marker) }
	case <-time.After(time.Second):
		t.Fatal("missing subject marker")
	}
	if ok, _ := ms.RemoveMsg(2); ok { t.Fatal("duplicate removal was not idempotent") }
}
