
import logging
import sys
from datetime import datetime
from typing import Any

LOGGER_NAME = "akton_agent"

logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)


def log_sql(message: str, sql: str | None = None, params: dict[str, Any] | None = None) -> None:
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    prefix = f"[SQL {ts}] {message}"
    if sql:
        logger.info("%s\n%s", prefix, sql)
    else:
        logger.info(prefix)
    if params:
        logger.info("params=%s", params)


def log_agent(message: str, **extra: Any) -> None:
    logger.info("AGENT: %s | extra=%s", message, extra)
