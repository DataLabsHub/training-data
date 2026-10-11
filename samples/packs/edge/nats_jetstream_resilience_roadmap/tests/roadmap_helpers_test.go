// Copyright 2026 The EdgeBench Authors
// Licensed under the Apache License, Version 2.0.

package server

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
)

func roadmapRunServer(t *testing.T, mutate func(*Options)) (*Server, *nats.Conn) {
	t.Helper()
	opts := &Options{Host: "127.0.0.1", Port: -1}
	opts.JetStream, opts.StoreDir, opts.NoSigs = true, t.TempDir(), true
	if mutate != nil {
		mutate(opts)
	}
	s, err := NewServer(opts)
	if err != nil {
		t.Fatal(err)
	}
	s.Start()
	if !s.ReadyForConnections(10 * time.Second) {
		s.Shutdown()
		t.Fatal("server not ready")
	}
	nc, err := nats.Connect(s.ClientURL(), nats.Timeout(2*time.Second))
	if err != nil {
		s.Shutdown()
		t.Fatal(err)
	}
	t.Cleanup(func() { nc.Close(); s.Shutdown(); s.WaitForShutdown() })
	return s, nc
}

func roadmapRequest(t *testing.T, nc *nats.Conn, subject string, request, response any) {
	t.Helper()
	data, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	msg, err := nc.Request(subject, data, 3*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(msg.Data, response); err != nil {
		t.Fatalf("decode %s: %v (%q)", subject, err, msg.Data)
	}
}

func roadmapAddStream(t *testing.T, nc *nats.Conn, cfg StreamConfig) *StreamInfo {
	t.Helper()
	var response JSApiStreamCreateResponse
	roadmapRequest(t, nc, fmt.Sprintf(JSApiStreamCreateT, cfg.Name), cfg, &response)
	if response.Error != nil {
		t.Fatalf("create stream: %v", response.Error)
	}
	return response.StreamInfo
}

func roadmapAddConsumer(t *testing.T, nc *nats.Conn, stream string, cfg ConsumerConfig) *JSApiConsumerCreateResponse {
	t.Helper()
	request := CreateConsumerRequest{Stream: stream, Config: cfg, Action: ActionCreate}
	var response JSApiConsumerCreateResponse
	roadmapRequest(t, nc, fmt.Sprintf(JSApiDurableCreateT, stream, cfg.Durable), request, &response)
	return &response
}

func roadmapEventually(t *testing.T, timeout time.Duration, check func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if check() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !check() {
		t.Fatal("condition not met before deadline")
	}
}

func roadmapDirectRequest(t *testing.T, nc *nats.Conn, stream string, req JSApiMsgGetRequest) *nats.Subscription {
	t.Helper()
	reply := nats.NewInbox()
	sub, err := nc.SubscribeSync(reply)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(req)
	if err := nc.PublishRequest(fmt.Sprintf("$JS.API.DIRECT.GET.%s", stream), reply, data); err != nil {
		t.Fatal(err)
	}
	return sub
}
