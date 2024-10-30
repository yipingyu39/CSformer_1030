import os
import logging
from datetime import datetime

class TrainingLogger:
    def __init__(self, round, log_dir="logs"):
        self.log_dir = log_dir
        self.log_file = self._create_log_file(round)
        self.logger = self._setup_logger()

    def _create_log_file(self,round):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"training_log_{timestamp}_{round}.log")
        return log_file

    def _setup_logger(self):
        logger = logging.getLogger('training_logger')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def log(self, message):
        self.logger.info(message)