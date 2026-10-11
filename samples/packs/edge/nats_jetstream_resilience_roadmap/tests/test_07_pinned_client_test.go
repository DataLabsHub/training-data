package server

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
)

func TestRoadmapPhase07(t *testing.T) {
	_, nc := roadmapRunServer(t, nil)
	roadmapAddStream(t, nc, StreamConfig{Name: "PINNED", Subjects: []string{"pin"}, Storage: MemoryStorage})
	resp := roadmapAddConsumer(t, nc, "PINNED", ConsumerConfig{Durable: "C", AckPolicy: AckExplicit, PriorityPolicy: PriorityPinnedClient, PriorityGroups: []string{"A"}, PinnedTTL: 150 * time.Millisecond})
	if resp.Error != nil { t.Fatal(resp.Error) }
	for i := 0; i < 2; i++ { if err := nc.Publish("pin", []byte("x")); err != nil { t.Fatal(err) } }
	inbox := nats.NewInbox(); sub, _ := nc.SubscribeSync(inbox)
	pull := func(id string) *nats.Msg {
		req, _ := json.Marshal(JSApiConsumerGetNextRequest{Batch: 1, Expires: time.Second, PriorityGroup: PriorityGroup{Group: "A", Id: id}})
		if err := nc.PublishRequest(fmt.Sprintf(JSApiRequestNextT, "PINNED", "C"), inbox, req); err != nil { t.Fatal(err) }
		msg, err := sub.NextMsg(time.Second); if err != nil { t.Fatal(err) }; return msg
	}
	first := pull("")
	pin := first.Header.Get(JSPullRequestNatsPinId)
	if pin == "" { t.Fatal("server did not assign pin id") }
	wrong := pull("wrong")
	if wrong.Header.Get("Status") != "423" { t.Fatalf("wrong pin status = %q", wrong.Header.Get("Status")) }
	time.Sleep(220 * time.Millisecond)
	second := pull("")
	if next := second.Header.Get(JSPullRequestNatsPinId); next == "" || next == pin { t.Fatalf("pin did not rotate: old=%q new=%q", pin, next) }
}
