"""Keep feature tests collectible when the base revision lacks the new module.

The real module is used whenever it is present (including the gold revision).  On
the base revision, placeholders let pytest execute the assertions and report real
test failures instead of converting every target into a collection-time MISSING
result.
"""

from enum import Enum
from types import ModuleType, SimpleNamespace
from pathlib import Path
import sys


if not Path(__file__).with_name("IndicatorEnrichment.py").is_file():
    indicator_enrichment = ModuleType("IndicatorEnrichment")

    class IndicatorType(Enum):
        IP = "IP"
        URL = "URL"
        DOMAIN = "DOMAIN"
        CVE = "CVE"
        FILE = "FILE"

        @classmethod
        def resolve_from_string(cls, value):
            return object()

    class UnsupportedIndicator:
        def __init__(self, type, value):
            self.type = type
            self.value = value

    class ValidationError(Exception):
        pass

    class AllExecutionsFailedError(Exception):
        pass

    class GracefulExit(Exception):
        pass

    class EnrichmentRequest:
        def __init__(
            self,
            valid_indicators_by_type,
            unsupported_items,
            unknown_items,
            duplicates_removed_count,
            sub_command_arguments,
            include_raw_context,
        ):
            self.valid_indicators_by_type = valid_indicators_by_type
            self.unsupported_items = unsupported_items
            self.unknown_items = unknown_items
            self.duplicates_removed_count = duplicates_removed_count
            self.sub_command_arguments = sub_command_arguments
            self.include_raw_context = include_raw_context

        @property
        def total_valid_count(self):
            return sum(len(values) for values in self.valid_indicators_by_type.values())

    class EnrichmentResult:
        def __init__(self):
            self.enriched_data = []
            self.raw_context = {}
            self.markdown_sections = []
            self.execution_errors = []

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            raise RuntimeError("IndicatorEnrichment is absent from the base revision")

        def format(self, *args, **kwargs):
            raise RuntimeError("IndicatorEnrichment is absent from the base revision")

        def build_and_validate(self, *args, **kwargs):
            raise RuntimeError("IndicatorEnrichment is absent from the base revision")

    class BatchExecutor:
        def execute_batch(self, *args, **kwargs):
            raise RuntimeError("IndicatorEnrichment is absent from the base revision")

    def main():
        raise RuntimeError("IndicatorEnrichment is absent from the base revision")

    indicator_enrichment.IndicatorType = IndicatorType
    indicator_enrichment.UnsupportedIndicator = UnsupportedIndicator
    indicator_enrichment.ValidationError = ValidationError
    indicator_enrichment.AllExecutionsFailedError = AllExecutionsFailedError
    indicator_enrichment.GracefulExit = GracefulExit
    indicator_enrichment.EnrichmentRequest = EnrichmentRequest
    indicator_enrichment.EnrichmentResult = EnrichmentResult
    indicator_enrichment.ResponseFormatter = _Unavailable
    indicator_enrichment.EnrichmentService = _Unavailable
    indicator_enrichment.EnrichmentRequestBuilder = _Unavailable
    indicator_enrichment.BatchExecutor = BatchExecutor
    indicator_enrichment.argToList = lambda value, transform=None: []
    indicator_enrichment.auto_detect_indicator_type = lambda value: None
    indicator_enrichment.execute_command = lambda *args, **kwargs: []
    indicator_enrichment.demisto = SimpleNamespace(args=lambda: {}, error=lambda *args, **kwargs: None)
    indicator_enrichment.return_results = lambda *args, **kwargs: None
    indicator_enrichment.return_error = lambda *args, **kwargs: None
    indicator_enrichment.main = main
    sys.modules["IndicatorEnrichment"] = indicator_enrichment
