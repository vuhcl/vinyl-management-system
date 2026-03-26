"""
grader/tests/test_harmonize.py
"""

import json
from pathlib import Path

import pytest
import yaml

from grader.src.data.harmonize_labels import LabelHarmonizer


@pytest.fixture
def harmonizer(test_config, guidelines_path):
    return LabelHarmonizer(
        config_path=test_config,
        guidelines_path=guidelines_path,
    )


@pytest.fixture
def discogs_jsonl(tmp_dirs, sample_unified_records):
    discogs_records = [
        r for r in sample_unified_records if r["source"] == "discogs"
    ]
    path = tmp_dirs["processed"] / "discogs_processed.jsonl"
    with open(path, "w") as f:
        for r in discogs_records:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def ebay_jsonl(tmp_dirs, sample_unified_records):
    ebay_records = [
        r for r in sample_unified_records if r["source"] == "ebay_jp"
    ]
    path = tmp_dirs["processed"] / "ebay_processed.jsonl"
    with open(path, "w") as f:
        for r in ebay_records:
            f.write(json.dumps(r) + "\n")
    return path


class TestSchemaValidation:
    def test_valid_record_passes(self, harmonizer, sample_unified_records):
        result = harmonizer.validate_record(sample_unified_records[0])
        assert result is None

    def test_missing_text_fails(self, harmonizer, sample_unified_records):
        record = {**sample_unified_records[0], "text": ""}
        result = harmonizer.validate_record(record)
        assert result is not None
        assert "text" in result

    def test_missing_item_id_fails(self, harmonizer, sample_unified_records):
        record = {
            k: v for k, v in sample_unified_records[0].items() if k != "item_id"
        }
        result = harmonizer.validate_record(record)
        assert result is not None

    def test_null_sleeve_label_fails(self, harmonizer, sample_unified_records):
        record = {**sample_unified_records[0], "sleeve_label": None}
        result = harmonizer.validate_record(record)
        assert result is not None


class TestGradeValidation:
    def test_valid_grades_pass(self, harmonizer, sample_unified_records):
        for record in sample_unified_records:
            result = harmonizer.validate_grades(record)
            assert result is None, (
                f"Valid record failed: {record['sleeve_label']}/"
                f"{record['media_label']} — {result}"
            )

    def test_invalid_sleeve_grade_fails(
        self, harmonizer, sample_unified_records
    ):
        record = {**sample_unified_records[0], "sleeve_label": "Excellent Plus"}
        result = harmonizer.validate_grades(record)
        assert result is not None
        assert "sleeve" in result

    def test_invalid_media_grade_fails(
        self, harmonizer, sample_unified_records
    ):
        record = {**sample_unified_records[0], "media_label": "Unknown"}
        result = harmonizer.validate_grades(record)
        assert result is not None

    def test_generic_as_media_fails(self, harmonizer, sample_unified_records):
        record = {**sample_unified_records[0], "media_label": "Generic"}
        result = harmonizer.validate_grades(record)
        # Should return generic_as_media_grade specifically
        # Requires the Generic check to come before the general media check
        # in validate_grades() — see fix in harmonize_labels.py
        assert result == "generic_as_media_grade"

    def test_generic_as_sleeve_passes(self, harmonizer, sample_unified_records):
        record = {**sample_unified_records[0], "sleeve_label": "Generic"}
        result = harmonizer.validate_grades(record)
        assert result is None


class TestDeduplication:
    def test_no_duplicates_unchanged(self, harmonizer, sample_unified_records):
        # Initialize stats before calling deduplicate
        harmonizer._stats = {
            "total_fetched": 0,
            "total_dropped": 0,
            "total_saved": 0,
            "per_source": {},
            "drops": {},
            "duplicates_removed": {},
            "cross_source_duplicates": 0,
        }
        discogs = [
            r for r in sample_unified_records if r["source"] == "discogs"
        ]
        deduped = harmonizer.deduplicate(discogs, "discogs")
        assert len(deduped) == len(discogs)

    def test_duplicate_item_ids_removed(
        self, harmonizer, sample_unified_records
    ):
        harmonizer._stats = {
            "total_fetched": 0,
            "total_dropped": 0,
            "total_saved": 0,
            "per_source": {},
            "drops": {},
            "duplicates_removed": {},
            "cross_source_duplicates": 0,
        }
        record = sample_unified_records[0]
        records_with_dup = [record, record]
        deduped = harmonizer.deduplicate(records_with_dup, "discogs")
        assert len(deduped) == 1

    def test_cross_source_dedup_keeps_discogs(
        self, harmonizer, sample_unified_records
    ):
        harmonizer._stats = {
            "total_fetched": 0,
            "total_dropped": 0,
            "total_saved": 0,
            "per_source": {},
            "drops": {},
            "duplicates_removed": {},
            "cross_source_duplicates": 0,
        }
        discogs_record = {
            **sample_unified_records[0],
            "source": "discogs",
            "artist": "Miles Davis",
            "title": "Kind of Blue",
        }
        ebay_record = {
            **sample_unified_records[0],
            "item_id": "ebay_999",
            "source": "ebay_jp",
            "artist": "Miles Davis",
            "title": "Kind of Blue",
        }
        deduped = harmonizer.deduplicate_cross_source(
            [discogs_record, ebay_record]
        )
        assert len(deduped) == 1
        assert deduped[0]["source"] == "discogs"


class TestDistributionReport:
    def test_distribution_counts_all_records(
        self, harmonizer, sample_unified_records
    ):
        dist = harmonizer.compute_distribution(sample_unified_records)
        sleeve_total = sum(dist["sleeve"].values())
        assert sleeve_total == len(sample_unified_records)

    def test_generic_only_in_sleeve(self, harmonizer, sample_unified_records):
        dist = harmonizer.compute_distribution(sample_unified_records)
        assert "Generic" in dist["sleeve"]
        assert "Generic" not in dist["media"]

    def test_rare_class_warning_fires(self, harmonizer, sample_unified_records):
        dist = {"sleeve": {"Poor": 1}, "media": {"Poor": 1}}
        warnings = harmonizer.flag_rare_classes(dist)
        assert len(warnings) > 0
        assert any("Poor" in w for w in warnings)


class TestRunMethod:
    def test_run_dry_produces_records(
        self, harmonizer, discogs_jsonl, ebay_jsonl
    ):
        records = harmonizer.run(dry_run=True)
        assert len(records) > 0

    def test_all_output_records_have_valid_grades(
        self, harmonizer, discogs_jsonl, ebay_jsonl, guidelines_path
    ):
        guidelines = yaml.safe_load(open(guidelines_path))
        sleeve_grades = set(guidelines["sleeve_grades"])
        media_grades = set(guidelines["media_grades"])

        records = harmonizer.run(dry_run=True)
        for record in records:
            assert record["sleeve_label"] in sleeve_grades
            assert record["media_label"] in media_grades
            assert record["media_label"] != "Generic"


class TestExcludeThinNotes:
    def test_exclude_thin_notes_drops_inadequate(
        self, tmp_dirs, test_config, guidelines_path
    ):
        processed = tmp_dirs["processed"]
        thin = {
            "item_id": "thin_1",
            "source": "discogs",
            "text": "unplayed",
            "sleeve_label": "Near Mint",
            "media_label": "Near Mint",
            "label_confidence": 1.0,
        }
        adequate = {
            "item_id": "ok_1",
            "source": "discogs",
            "text": (
                "jacket has corner wear and ring wear, "
                "vinyl plays cleanly with no pops"
            ),
            "sleeve_label": "Very Good",
            "media_label": "Near Mint",
            "label_confidence": 1.0,
        }
        with open(processed / "discogs_processed.jsonl", "w") as f:
            f.write(json.dumps(thin) + "\n")
            f.write(json.dumps(adequate) + "\n")
        with open(processed / "ebay_processed.jsonl", "w") as f:
            pass

        cfg_path = Path(test_config)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["harmonization"]["exclude_thin_notes"] = True
        thin_cfg = cfg_path.parent / "harmonize_exclude_thin.yaml"
        with open(thin_cfg, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        h = LabelHarmonizer(
            config_path=str(thin_cfg),
            guidelines_path=guidelines_path,
        )
        records = h.run(dry_run=True)
        assert len(records) == 1
        assert records[0]["item_id"] == "ok_1"
        assert h._stats["drops"].get("thin_note_inadequate") == 1
