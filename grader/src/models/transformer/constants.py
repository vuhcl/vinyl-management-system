"""Shared symbols for DistilBERT transformer training."""

TARGETS = ["sleeve", "media"]
SPLITS = ["train", "val", "test"]

EVIDENCE_STRENGTH_TO_IDX: dict[str, int] = {
    "none": 0,
    "weak": 1,
    "strong": 2,
}
_META_HPARAM_KEYS = frozenset({"description", "name"})
