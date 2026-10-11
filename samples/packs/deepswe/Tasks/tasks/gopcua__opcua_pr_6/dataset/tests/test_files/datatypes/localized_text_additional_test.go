package datatypes

import "testing"

func TestAdditionalLocalizedTextMasksAndRoundTrip(t *testing.T) {
	cases := []struct {
		name       string
		locale     string
		text       string
		wantMask   uint8
		wantLocale bool
		wantText   bool
	}{
		{"empty", "", "", 0x00, false, false},
		{"locale-only", "en-US", "", 0x01, true, false},
		{"text-only", "", "Server", 0x02, false, true},
		{"both", "ja-JP", "Endpoint", 0x03, true, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value := NewLocalizedText(tc.locale, tc.text)
			if value.EncodingMask != tc.wantMask {
				t.Fatalf("EncodingMask = %#x, want %#x", value.EncodingMask, tc.wantMask)
			}

			encoded, err := value.Serialize()
			if err != nil {
				t.Fatalf("Serialize failed: %v", err)
			}
			decoded, err := DecodeLocalizedText(encoded)
			if err != nil {
				t.Fatalf("DecodeLocalizedText failed: %v", err)
			}

			if decoded.HasLocale() != tc.wantLocale || decoded.HasText() != tc.wantText {
				t.Fatalf("decoded mask locale/text = %v/%v, want %v/%v", decoded.HasLocale(), decoded.HasText(), tc.wantLocale, tc.wantText)
			}
			if tc.wantLocale && decoded.Locale.Get() != tc.locale {
				t.Fatalf("Locale = %q, want %q", decoded.Locale.Get(), tc.locale)
			}
			if tc.wantText && decoded.Text.Get() != tc.text {
				t.Fatalf("Text = %q, want %q", decoded.Text.Get(), tc.text)
			}
		})
	}
}

func TestAdditionalStringArrayNilAndEmptyEncoding(t *testing.T) {
	nilArray := NewStringArray(nil)
	if nilArray.ArraySize != -1 {
		t.Fatalf("nil array size = %d, want -1", nilArray.ArraySize)
	}
	nilBytes, err := nilArray.Serialize()
	if err != nil {
		t.Fatalf("Serialize nil array failed: %v", err)
	}
	decodedNil, err := DecodeStringArray(nilBytes)
	if err != nil {
		t.Fatalf("Decode nil array failed: %v", err)
	}
	if decodedNil.ArraySize != -1 || len(decodedNil.Strings) != 0 {
		t.Fatalf("decoded nil array size/items = %d/%d, want -1/0", decodedNil.ArraySize, len(decodedNil.Strings))
	}

	emptyArray := NewStringArray([]string{})
	if emptyArray.ArraySize != 0 {
		t.Fatalf("empty array size = %d, want 0", emptyArray.ArraySize)
	}
	emptyBytes, err := emptyArray.Serialize()
	if err != nil {
		t.Fatalf("Serialize empty array failed: %v", err)
	}
	decodedEmpty, err := DecodeStringArray(emptyBytes)
	if err != nil {
		t.Fatalf("Decode empty array failed: %v", err)
	}
	if decodedEmpty.ArraySize != 0 || len(decodedEmpty.Strings) != 0 {
		t.Fatalf("decoded empty array size/items = %d/%d, want 0/0", decodedEmpty.ArraySize, len(decodedEmpty.Strings))
	}
}
