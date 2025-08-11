import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Environment
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Audio processing settings
    SAMPLE_RATE = 22050
    HOP_LENGTH = 512
    N_FFT = 2048
    N_MELS = 128
    
    # File constraints
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 25))
    MAX_FILES_PER_BATCH = int(os.getenv('MAX_FILES_PER_BATCH', 15))
    MIN_FILES_PER_BATCH = 3
    SUPPORTED_FORMATS = ['mp3', 'wav', 'flac', 'm4a', 'ogg']
    
    # Model settings
    MODEL_VERSION = "2.0"
    CONFIDENCE_THRESHOLD = 0.7
    
    # Feature extraction settings
    FEATURE_DIMENSIONS = 47
    
    # API Keys (optional)
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', None)
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    TEMP_DIR = PROJECT_ROOT / "temp"
    
    # Report settings
    REPORT_TEMPLATE = "professional"
    INCLUDE_TECHNICAL_DETAILS = True
    
    # UI Settings
    THEME = "dark"
    SHOW_ADVANCED_METRICS = True
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        # Create necessary directories
        for dir_attr in ['DATA_DIR', 'MODELS_DIR', 'REPORTS_DIR', 'TEMP_DIR']:
            dir_path = getattr(cls, dir_attr)
            dir_path.mkdir(exist_ok=True, parents=True)
        
        return True

# Create config instance
config = Config()
config.validate()

def get_project_dirs():
    """Get project directories"""
    return {
        'root': config.PROJECT_ROOT,
        'data': config.DATA_DIR,
        'models': config.MODELS_DIR,
        'reports': config.REPORTS_DIR,
        'temp': config.TEMP_DIR
    }
