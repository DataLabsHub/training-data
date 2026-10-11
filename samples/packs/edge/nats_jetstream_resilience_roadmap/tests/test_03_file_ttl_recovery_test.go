package server

import (
	"testing"
	"time"
)

func TestRoadmapPhase03(t *testing.T) {
	dir := t.TempDir()
	cfg := StreamConfig{Name: "FILETTL", Subjects: []string{"events"}, Storage: FileStorage, AllowMsgTTL: true}
	fs, err := newFileStore(FileStoreConfig{StoreDir: dir}, cfg)
	if err != nil { t.Fatal(err) }
	if _, _, err := fs.StoreMsg("events", []byte("NATS/1.0\r\nNats-TTL: 1s\r\n\r\n"), []byte("one"), 1); err != nil { t.Fatal(err) }
	if err := fs.Stop(); err != nil { t.Fatal(err) }
	fs, err = newFileStore(FileStoreConfig{StoreDir: dir}, cfg)
	if err != nil { t.Fatal(err) }
	defer fs.Stop()
	if state := fs.State(); state.Msgs != 1 { t.Fatalf("restart lost live message: %+v", state) }
	roadmapEventually(t, 3*time.Second, func() bool { return fs.State().Msgs == 0 })
	if state := fs.State(); state.FirstSeq != 2 || state.LastSeq != 1 { t.Fatalf("expired state = %+v", state) }
}
