package server

import (
	"testing"
	"time"
)

func TestRoadmapPhase06(t *testing.T) {
	_, nc := roadmapRunServer(t, nil)
	roadmapAddStream(t, nc, StreamConfig{Name: "GROUPS", Subjects: []string{"g.>"}, Storage: MemoryStorage})
	valid := roadmapAddConsumer(t, nc, "GROUPS", ConsumerConfig{Durable: "PIN", AckPolicy: AckExplicit, PriorityPolicy: PriorityPinnedClient, PriorityGroups: []string{"gold"}, PinnedTTL: time.Second})
	if valid.Error != nil || valid.ConsumerInfo.Config.PriorityGroups[0] != "gold" { t.Fatalf("valid group: %+v", valid) }
	missing := roadmapAddConsumer(t, nc, "GROUPS", ConsumerConfig{Durable: "MISSING", AckPolicy: AckExplicit, PriorityPolicy: PriorityOverflow})
	if missing.Error == nil || missing.Error.ErrCode != uint16(JSConsumerPriorityPolicyWithoutGroup) { t.Fatalf("missing group accepted: %+v", missing) }
	empty := roadmapAddConsumer(t, nc, "GROUPS", ConsumerConfig{Durable: "EMPTY", AckPolicy: AckExplicit, PriorityPolicy: PriorityOverflow, PriorityGroups: []string{""}})
	if empty.Error == nil || empty.Error.ErrCode != uint16(JSConsumerEmptyGroupName) { t.Fatalf("empty group accepted: %+v", empty) }
}
