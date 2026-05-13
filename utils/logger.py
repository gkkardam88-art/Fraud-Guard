"""
Centralised logging.
Every module does:  log = logging.getLogger(__name__)
Call setup_logging() once at process startup.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB per file
    backup_count: int = 3,
) -> None:
    fmt = "%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    for h in handlers:
        h.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers we don't control
    for lib in ("httpx", "httpcore", "openai._base_client", "kafka", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)
