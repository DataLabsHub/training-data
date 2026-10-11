package server

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
)

func TestRoadmapPhase08(t *testing.T) {
	_, nc := roadmapRunServer(t, nil)
	roadmapAddStream(t, nc, StreamConfig{Name: "OVER", Subjects: []string{"over"}, Storage: MemoryStorage})
	resp := roadmapAddConsumer(t, nc, "OVER", ConsumerConfig{Durable: "C", AckPolicy: AckExplicit, PriorityPolicy: PriorityOverflow, PriorityGroups: []string{"A"}})
	if resp.Error != nil { t.Fatal(resp.Error) }
	for i := 0; i < 5; i++ { if err := nc.Publish("over", []byte("x")); err != nil { t.Fatal(err) } }
	if err := nc.Flush(); err != nil { t.Fatal(err) }
	pull := func(min int64) *nats.Msg {
		inbox := nats.NewInbox(); sub, _ := nc.SubscribeSync(inbox); defer sub.Unsubscribe()
		req, _ := json.Marshal(JSApiConsumerGetNextRequest{Batch: 1, Expires: 120 * time.Millisecond, PriorityGroup: PriorityGroup{Group: "A", MinPending: min}})
		if err := nc.PublishRequest(fmt.Sprintf(JSApiRequestNextT, "OVER", "C"), inbox, req); err != nil { t.Fatal(err) }
		msg, err := sub.NextMsg(time.Second); if err != nil { t.Fatal(err) }; return msg
	}
	if msg := pull(10); msg.Header.Get("Status") != "408" { t.Fatalf("below threshold = %q", msg.Header.Get("Status")) }
	if msg := pull(1); len(msg.Data) == 0 || msg.Header.Get("Status") != "" { t.Fatalf("above threshold = status %q data %q", msg.Header.Get("Status"), msg.Data) }
}
