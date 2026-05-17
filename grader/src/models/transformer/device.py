"""Compute device selection (MPS / CUDA / CPU)."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Select compute device with MPS priority for Apple Silicon.
    Falls back to CUDA then CPU.
    """
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon) for training.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA for training.")
        return torch.device("cuda")
    else:
        logger.info("Using CPU for training.")
        return torch.device("cpu")
