---
name: Grader Tier C rubric change (sign-off)
description: Schema / label-space / map changes to grading_guidelines.yaml — stakeholder review before merge
title: "[Tier C] "
labels: []
---

## Summary

Describe the **canonical** grade or map change (sleeve/media lists, `discogs_condition_map`, `ebay_jp_harmonization`, etc.).

## Stakeholder sign-off

- [ ] Product or domain owner has reviewed the semantic impact on sleeve vs media.
- [ ] Dataset policy chosen for mixed-era labels: accept mix / filter / backfill (document in PR).

## Engineering checklist

- [ ] `guidelines_version` in `grading_guidelines.yaml` will be bumped in the same PR.
- [ ] `test_guidelines.py` / blast-radius updates planned (`tfidf_features`, harmonization, encoders if label count changes).
- [ ] Committed `grader/reports/rule_engine_baseline.json` updated or regenerated with matching `guidelines_version`.
