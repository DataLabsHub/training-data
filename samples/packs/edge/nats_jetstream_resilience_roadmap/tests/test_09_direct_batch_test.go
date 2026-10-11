package server

import (
	"testing"
	"time"
)

func TestRoadmapPhase09(t *testing.T) {
	_, nc := roadmapRunServer(t, nil)
	roadmapAddStream(t, nc, StreamConfig{Name: "BATCH", Subjects: []string{"b.*"}, Storage: MemoryStorage, AllowDirect: true})
	for _, subject := range []string{"b.a", "b.b", "b.c", "b.d", "b.e"} {
		if _, err := nc.Request(subject, []byte(subject), time.Second); err != nil { t.Fatal(err) }
	}
	sub := roadmapDirectRequest(t, nc, "BATCH", JSApiMsgGetRequest{Seq: 1, Batch: 3})
	defer sub.Unsubscribe()
	for i, subject := range []string{"b.a", "b.b", "b.c"} {
		msg, err := sub.NextMsg(time.Second); if err != nil { t.Fatal(err) }
		if msg.Header.Get(JSSubject) != subject || msg.Header.Get(JSSequence) == "" { t.Fatalf("message %d headers = %v", i, msg.Header) }
	}
	eob, err := sub.NextMsg(time.Second)
	if err != nil || eob.Header.Get("Status") != "204" || eob.Header.Get("Description") != "EOB" { t.Fatalf("EOB = %v, %v", eob, err) }
}
