package server

import (
	"sort"
	"testing"
	"time"
)

func TestRoadmapPhase10(t *testing.T) {
	_, nc := roadmapRunServer(t, nil)
	roadmapAddStream(t, nc, StreamConfig{Name: "MULTI", Subjects: []string{"m.*"}, Storage: MemoryStorage, AllowDirect: true})
	for round := 0; round < 2; round++ {
		for _, subject := range []string{"m.a", "m.b", "m.c"} { if _, err := nc.Request(subject, []byte{byte('0' + round)}, time.Second); err != nil { t.Fatal(err) } }
	}
	sub := roadmapDirectRequest(t, nc, "MULTI", JSApiMsgGetRequest{MultiLastFor: []string{"m.*"}, UpToSeq: 3})
	defer sub.Unsubscribe()
	var got []string
	for i := 0; i < 3; i++ { msg, err := sub.NextMsg(time.Second); if err != nil { t.Fatal(err) }; got = append(got, msg.Header.Get(JSSubject)); if string(msg.Data) != "0" { t.Fatalf("up-to bound ignored: %q", msg.Data) } }
	sort.Strings(got)
	if want := []string{"m.a", "m.b", "m.c"}; len(got) != len(want) || got[0] != want[0] || got[2] != want[2] { t.Fatalf("subjects = %v", got) }
	eob, err := sub.NextMsg(time.Second); if err != nil || eob.Header.Get("Description") != "EOB" { t.Fatalf("EOB = %v, %v", eob, err) }
	none := roadmapDirectRequest(t, nc, "MULTI", JSApiMsgGetRequest{MultiLastFor: []string{"none.*"}})
	msg, err := none.NextMsg(time.Second); if err != nil || msg.Header.Get("Status") != "404" { t.Fatalf("no results = %v, %v", msg, err) }
}
