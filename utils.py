import os
import logging

from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

load_dotenv(dotenv_path='.env')



PATH_TREEBANKS = os.getenv('PATH_TREEBANKS')
PATH_MODELS = os.getenv('PATH_MODELS')
PATH_BERTFORDEPREL_VENV = os.getenv('PATH_BERTFORDEPREL_VENV')
PATH_BERTFORDEPREL_SCRIPT = os.getenv('PATH_BERTFORDEPREL_SCRIPT')
LOGS_DIR = os.path.join(PATH_TREEBANKS, 'logs')



def setup_logging():
    if not logging.getLogger().hasHandlers():
        log_file = os.path.join(LOGS_DIR, 'training.log')
        handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
