import argparse
import uvicorn
import sys
from src.core.config import settings, SystemMode
from src.infra.logging import setup_logging, get_logger
from src.infra.database import init_db

setup_logging()
logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LCD Inspection System")
    parser.add_argument("--mode", type=str, choices=["production", "developer"], help="Override system mode")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (Developer Mode only)")
    parser.add_argument("--headless", action="store_true", help="Force headless mode")
    return parser.parse_args()

def main():
    args = parse_args()

    # Override settings if provided via CLI
    if args.mode:
        settings.MODE = SystemMode(args.mode)

    logger.info(f"System starting in {settings.MODE} mode")

    # Initialize DB
    logger.info("Initializing Database...")
    init_db()
    logger.info(f"Database initialized at {settings.DB_PATH}")

    if args.gui:
        if settings.MODE != SystemMode.DEVELOPER:
             logger.warning("GUI flag ignored because system is not in DEVELOPER mode.")
        else:
             logger.info("Launching GUI...")
             # Adjust sys.path or import logic if needed, but src package should work from root
             from src.gui.main_window import main as gui_main
             gui_main()
             return

    # Default: Start API (Headless)
    logger.info("Starting API Server...")
    # Import app here to avoid circular or early init issues if any
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
