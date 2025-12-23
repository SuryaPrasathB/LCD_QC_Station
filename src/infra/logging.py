import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from src.core.config import settings

class JSONFormatter(logging.Formatter):
    """Formats log records as a JSON object."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        # Add any extra attributes
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)

def setup_logging():
    """Configures system-wide logging: JSON to file, Human-readable to Console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove default handlers
    root_logger.handlers = []

    # 1. File Handler (JSON)
    log_file = settings.LOG_DIR / "system.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # 2. Console Handler (Human Readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO) # Can be DEBUG in dev
    root_logger.addHandler(console_handler)

    logging.info("Logging initialized.")

def get_logger(name: str):
    return logging.getLogger(name)
