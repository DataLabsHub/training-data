package server

import "testing"

func TestRoadmapPhase01(t *testing.T) {
	for value, want := range map[string]int64{"1s": 1, "1500ms": 1, "never": -1, "2m": 120} {
		got, err := parseMessageTTL(value)
		if err != nil || got != want {
			t.Fatalf("parse %q = %d, %v; want %d", value, got, err, want)
		}
	}
	for _, value := range []string{"", "0s", "500ms", "-1s", "bogus"} {
		if _, err := parseMessageTTL(value); err == nil {
			t.Fatalf("expected %q to be rejected", value)
		}
	}
	hdr := []byte("NATS/1.0\r\nNats-TTL: 9s\r\n\r\n")
	if ttl, err := getMessageTTL(hdr); err != nil || ttl != 9 {
		t.Fatalf("header ttl = %d, %v", ttl, err)
	}
}
