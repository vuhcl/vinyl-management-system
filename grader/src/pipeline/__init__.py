"""
grader/src/pipeline/

Top-level orchestrator for the vinyl condition grader.
Exposes two distinct pipelines:

  1. Training pipeline  — end-to-end: ingest → preprocess →
                          features → baseline → transformer →
                          compare → calibration → rule coverage

  2. Inference pipeline — single or batch: raw text →
                          preprocess → model predict →
                          rule engine → final prediction

All preprocessing is handled internally at inference time.
Callers (iOS app, CLI, batch job) pass raw text only.

Usage:
    # Training
    python -m grader.src.pipeline train
    python -m grader.src.pipeline train --skip-ingest
    python -m grader.src.pipeline train --skip-sale-history
    python -m grader.src.pipeline train --baseline-only

    # Inference
    python -m grader.src.pipeline predict --text "NM sleeve, plays perfectly"
    python -m grader.src.pipeline predict --file texts.txt
"""

from .model import Pipeline

__all__ = ["Pipeline"]
