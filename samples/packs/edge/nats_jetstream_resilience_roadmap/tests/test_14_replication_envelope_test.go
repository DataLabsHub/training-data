package server

import (
	"bytes"
	"testing"
)

func TestRoadmapPhase14(t *testing.T) {
	hdr, body := []byte("NATS/1.0\r\nNats-TTL: 5s\r\n\r\n"), []byte("payload")
	for _, sourced := range []bool{false, true} {
		encoded := encodeStreamMsg("orders.eu", "reply", hdr, body, 42, 99, sourced)
		subj, reply, gotHdr, gotBody, seq, ts, gotSourced, err := decodeStreamMsg(encoded[1:])
		if err != nil || subj != "orders.eu" || reply != "reply" || seq != 42 || ts != 99 || gotSourced != sourced || !bytes.Equal(gotHdr, hdr) || !bytes.Equal(gotBody, body) { t.Fatalf("round trip sourced=%v: %q %q %d %d %v %v", sourced, subj, reply, seq, ts, gotSourced, err) }
	}
	if _, _, _, _, _, _, _, err := decodeStreamMsg([]byte("corrupt")); err == nil { t.Fatal("corrupt replication envelope accepted") }
}
