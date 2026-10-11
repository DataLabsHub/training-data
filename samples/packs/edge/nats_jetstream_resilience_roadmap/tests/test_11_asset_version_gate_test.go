package server

import (
	"strconv"
	"testing"
	"time"
)

func TestRoadmapPhase11(t *testing.T) {
	sc := &StreamConfig{Name: "V", AllowMsgTTL: true, ExpiryWorkBudget: 4, Metadata: map[string]string{"tenant": "blue", JSServerVersionMetadataKey: "spoof"}}
	setStaticStreamMetadata(sc)
	if sc.Metadata["tenant"] != "blue" || sc.Metadata[JSServerVersionMetadataKey] != "" { t.Fatalf("static metadata = %v", sc.Metadata) }
	level, err := strconv.Atoi(sc.Metadata[JSRequiredLevelMetadataKey]); if err != nil || level < 1 { t.Fatalf("required level = %q", sc.Metadata[JSRequiredLevelMetadataKey]) }
	dynamic := setDynamicStreamMetadata(sc)
	if dynamic.Metadata[JSServerVersionMetadataKey] != VERSION || dynamic.Metadata[JSServerLevelMetadataKey] != strconv.Itoa(JSApiLevel) { t.Fatalf("dynamic metadata = %v", dynamic.Metadata) }
	deadline := time.Now().Add(time.Hour)
	cc := &ConsumerConfig{PauseUntil: &deadline, PriorityPolicy: PriorityPinnedClient, PriorityGroups: []string{"A"}, Metadata: map[string]string{"owner": "ops"}}
	setStaticConsumerMetadata(cc)
	if cc.Metadata["owner"] != "ops" || cc.Metadata[JSRequiredLevelMetadataKey] != "1" { t.Fatalf("consumer metadata = %v", cc.Metadata) }
}
