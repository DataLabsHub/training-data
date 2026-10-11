package server

import (
	"encoding/json"
	"testing"
)

func TestRoadmapPhase12(t *testing.T) {
	req := JSApiLeaderStepdownRequest{Placement: &Placement{Cluster: "C", Preferred: "S2", Tags: []string{"ssd"}}}
	data, err := json.Marshal(req); if err != nil { t.Fatal(err) }
	var decoded JSApiLeaderStepdownRequest
	if err := json.Unmarshal(data, &decoded); err != nil { t.Fatal(err) }
	if decoded.Placement == nil || decoded.Placement.Preferred != "S2" || decoded.Placement.Cluster != "C" { t.Fatalf("round trip = %+v", decoded) }
	_, nc := roadmapRunServer(t, nil)
	var response JSApiStreamCreateResponse
	cfg := StreamConfig{Name: "PREFERRED", Subjects: []string{"p"}, Storage: MemoryStorage, Placement: &Placement{Preferred: "missing"}}
	roadmapRequest(t, nc, "$JS.API.STREAM.CREATE.PREFERRED", cfg, &response)
	if response.Error == nil { t.Fatal("standalone server accepted preferred clustered placement") }
}
