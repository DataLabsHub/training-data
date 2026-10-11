from collections import defaultdict

import pytest

from IndicatorEnrichment import IndicatorType, EnrichmentRequestBuilder, ValidationError


def test_builder_groups_and_deduplicates_auto_detected_indicators(monkeypatch):
    monkeypatch.setattr(
        "IndicatorEnrichment.auto_detect_indicator_type",
        lambda value: "IP" if value[0].isdigit() else "Domain",
    )
    request = EnrichmentRequestBuilder(
        {"indicator_list": ["1.1.1.1", "example.com", "1.1.1.1"], "additional_fields": "true"}
    ).build_and_validate()

    assert request.valid_indicators_by_type[IndicatorType.IP] == ["1.1.1.1"]
    assert request.valid_indicators_by_type[IndicatorType.DOMAIN] == ["example.com"]
    assert request.duplicates_removed_count == 1
    assert request.sub_command_arguments == {"additional_fields": "true"}


def test_builder_rejects_more_than_one_hundred_unique_indicators(monkeypatch):
    monkeypatch.setattr("IndicatorEnrichment.auto_detect_indicator_type", lambda value: "IP")
    with pytest.raises(ValidationError, match="limit is 100"):
        EnrichmentRequestBuilder({"indicator_list": [f"192.0.2.{i}" for i in range(1, 102)]}).build_and_validate()
