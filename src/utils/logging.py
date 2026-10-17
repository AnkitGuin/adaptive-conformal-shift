"""Minimal logging setup shared by all experiment scripts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "acp", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stderr and (optionally) to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in logger.handlers):
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(fh)
    logger.propagate = False
    return logger
