package server

import "testing"

func TestRoadmapPhase17(t *testing.T) {
	ms, err := newMemStore(&StreamConfig{Name: "COMPAT", Storage: MemoryStorage}); if err != nil { t.Fatal(err) }; defer ms.Stop()
	seq, _, err := ms.StoreMsg("legacy", nil, []byte("one")); if err != nil || seq != 1 { t.Fatalf("legacy StoreMsg = %d %v", seq, err) }
	if msg, err := ms.LoadMsg(1, nil); err != nil || string(msg.msg) != "one" { t.Fatalf("legacy LoadMsg = %v %v", msg, err) }
	if purged, err := ms.PurgeEx("legacy", 0, 0); err != nil || purged != 1 { t.Fatalf("legacy PurgeEx = %d %v", purged, err) }
	if removed, err := ms.RemoveMsg(1); err != nil || removed { t.Fatalf("duplicate remove = %v %v", removed, err) }
}
