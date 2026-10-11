package server

import (
	"testing"
)

func TestRoadmapPhase15(t *testing.T) {
	hdr := []byte("NATS/1.0\r\nNats-Trace-Dest: trace.events\r\ntraceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01\r\nX-App: edge\r\n\r\n")
	values, external := genHeaderMapIfTraceHeadersPresent(hdr)
	if external || values[MsgTraceDest][0] != "trace.events" || values[traceParentHdr][0] == "" || values["X-App"][0] != "edge" { t.Fatalf("explicit trace headers = %v external=%v", values, external) }
	externalHdr := []byte("NATS/1.0\r\ntraceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01\r\nX-App: edge\r\n\r\n")
	if values, external := genHeaderMapIfTraceHeadersPresent(externalHdr); !external || values[traceParentHdr][0] == "" { t.Fatalf("external trace = %v %v", values, external) }
	unsampled := []byte("NATS/1.0\r\ntraceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00\r\n\r\n")
	if values, external := genHeaderMapIfTraceHeadersPresent(unsampled); external || values != nil { t.Fatalf("unsampled trace accepted: %v %v", values, external) }
	c := &client{kind: ROUTER, route: &route{remoteName: "route-a"}}
	c.opts.Name = "connection-a"
	if got := getConnName(c); got != "route-a" { t.Fatalf("route name = %q", got) }
	c.route.remoteName = ""
	if got := getConnName(c); got != "connection-a" { t.Fatalf("fallback name = %q", got) }
}
