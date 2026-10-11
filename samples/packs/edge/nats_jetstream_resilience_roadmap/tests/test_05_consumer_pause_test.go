package server

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
)

func TestRoadmapPhase05(t *testing.T) {
	_, nc := roadmapRunServer(t, nil)
	roadmapAddStream(t, nc, StreamConfig{Name: "PAUSE", Subjects: []string{"pause"}, Storage: MemoryStorage})
	until := time.Now().Add(180 * time.Millisecond).UTC()
	resp := roadmapAddConsumer(t, nc, "PAUSE", ConsumerConfig{Durable: "C", AckPolicy: AckExplicit, PauseUntil: &until})
	if resp.Error != nil || resp.ConsumerInfo == nil || resp.ConsumerInfo.Config.PauseUntil == nil { t.Fatalf("pause create: %+v", resp) }
	if err := nc.Publish("pause", []byte("held")); err != nil { t.Fatal(err) }
	if err := nc.Flush(); err != nil { t.Fatal(err) }
	inbox := nats.NewInbox()
	sub, _ := nc.SubscribeSync(inbox)
	req, _ := json.Marshal(JSApiConsumerGetNextRequest{Batch: 1, Expires: time.Second})
	if err := nc.PublishRequest(fmt.Sprintf(JSApiRequestNextT, "PAUSE", "C"), inbox, req); err != nil { t.Fatal(err) }
	if _, err := sub.NextMsg(80 * time.Millisecond); err != nats.ErrTimeout { t.Fatalf("delivery was not paused: %v", err) }
	msg, err := sub.NextMsg(time.Second)
	if err != nil || string(msg.Data) != "held" { t.Fatalf("resume delivery: %v, %q", err, msg.Data) }
}
