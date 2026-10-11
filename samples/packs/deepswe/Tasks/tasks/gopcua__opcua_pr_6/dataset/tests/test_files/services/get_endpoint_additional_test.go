package services

import (
	"testing"
	"time"
)

func TestAdditionalGetEndpointRequestServiceDispatch(t *testing.T) {
	request := NewGetEndpointRequest(
		time.Date(2018, time.August, 10, 23, 0, 0, 0, time.UTC),
		7, 0, 2500, "audit-id",
		"opc.tcp://example.invalid:4840/UA/Server",
		[]string{"en-US", "de-DE"},
		[]string{"http://opcfoundation.org/UA-Profile/Transport/uatcp-uasc-uabinary"},
	)

	encoded, err := request.Serialize()
	if err != nil {
		t.Fatalf("Serialize request failed: %v", err)
	}
	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatalf("Decode service failed: %v", err)
	}

	got, ok := decoded.(*GetEndpointRequest)
	if !ok {
		t.Fatalf("Decode returned %T, want *GetEndpointRequest", decoded)
	}
	if got.ServiceType() != ServiceTypeGetEndpointRequest {
		t.Fatalf("ServiceType = %d, want %d", got.ServiceType(), ServiceTypeGetEndpointRequest)
	}
	if got.EndpointURL.Get() != "opc.tcp://example.invalid:4840/UA/Server" {
		t.Fatalf("EndpointURL = %q", got.EndpointURL.Get())
	}
	if got.LocalIDs.ArraySize != 2 || got.LocalIDs.Strings[1].Get() != "de-DE" {
		t.Fatalf("LocalIDs did not round trip")
	}
	if got.ProfileURIs.ArraySize != 1 || got.ProfileURIs.Strings[0].Get() == "" {
		t.Fatalf("ProfileURIs did not round trip")
	}
}

func TestAdditionalGetEndpointResponseRoundTripEndpoints(t *testing.T) {
	endpoint := NewEndpointDesctiption(
		"opc.tcp://example.invalid:4840/UA/Server",
		NewApplicationDescription(
			"urn:example:server", "urn:example:product", "Example Server",
			AppTypeServer, "", "", []string{"opc.tcp://example.invalid:4840"},
		),
		nil,
		SecModeNone,
		"http://opcfoundation.org/UA/SecurityPolicy#None",
		NewUserTokenPolicyArray([]*UserTokenPolicy{
			NewUserTokenPolicy("anonymous", UserTokenAnonymous, "", "", ""),
		}),
		"http://opcfoundation.org/UA-Profile/Transport/uatcp-uasc-uabinary",
		0,
	)
	response := NewGetEndpointResponse(
		time.Date(2018, time.August, 10, 23, 0, 0, 0, time.UTC),
		7, 0, NewNullDiagnosticInfo(), nil, endpoint,
	)

	encoded, err := response.Serialize()
	if err != nil {
		t.Fatalf("Serialize response failed: %v", err)
	}
	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatalf("Decode service failed: %v", err)
	}

	got, ok := decoded.(*GetEndpointResponse)
	if !ok {
		t.Fatalf("Decode returned %T, want *GetEndpointResponse", decoded)
	}
	if got.ServiceType() != ServiceTypeGetEndpointResponse {
		t.Fatalf("ServiceType = %d, want %d", got.ServiceType(), ServiceTypeGetEndpointResponse)
	}
	if got.Endpoints.ArraySize != 1 || len(got.Endpoints.EndpointDescriptions) != 1 {
		t.Fatalf("decoded endpoints size/items = %d/%d, want 1/1", got.Endpoints.ArraySize, len(got.Endpoints.EndpointDescriptions))
	}
	if got.Endpoints.EndpointDescriptions[0].Server.ApplicationName.Text.Get() != "Example Server" {
		t.Fatalf("server application name did not round trip")
	}
}
