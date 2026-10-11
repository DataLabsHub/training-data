from pathlib import Path


def test_adjacent_enrichment_command_metadata_remains_available():
    metadata = Path("Packs/AggregatedScripts/Scripts/FileEnrichment/FileEnrichment.yml").read_text(encoding="utf-8")
    assert "name: file-enrichment" in metadata
