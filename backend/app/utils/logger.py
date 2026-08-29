"""Structured JSON logging for auditable execution traces."""

from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


class _SatQueryFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("service", "satquery-backend")
        log_record.setdefault("logger", record.name)
        log_record.setdefault("level", record.levelname)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _SatQueryFormatter("%(asctime)s %(level)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
