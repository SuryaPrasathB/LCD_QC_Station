import os
from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class SystemMode(str, Enum):
    PRODUCTION = "production"
    DEVELOPER = "developer"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # System
    MODE: SystemMode = SystemMode.DEVELOPER
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"

    # Storage
    DB_PATH: str = "sqlite:///data/db/production.sqlite"
    TEST_DB_PATH: str = "sqlite:///data/db/test.sqlite"
    IMAGE_STORE_PATH: Path = DATA_DIR / "images"
    LOG_DIR: Path = DATA_DIR / "logs"
    DATASET_DIR: Path = DATA_DIR / "datasets"

    # Vision
    ALIGNMENT_MIN_KEYPOINTS: int = 10
    ALIGNMENT_CONFIDENCE_THRESHOLD: float = 0.6
    SCORE_PASS_THRESHOLD: float = 0.8

    # Camera
    CAMERA_TIMEOUT: int = 5 # seconds

# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.IMAGE_STORE_PATH, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)
os.makedirs(settings.DATASET_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR / "db", exist_ok=True)
