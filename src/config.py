import os
from datetime import datetime
import yaml


class Config:
    def __init__(self):
        # Current dir absolute path
        _base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

        # Load conf.yml into class properties
        with open(os.path.join(_base_dir, 'conf.yml'), 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        self.ENCODING = conf['encoding']
        self.EPOCHS = conf['epochs']
        self.BATCH_SIZE = conf['batch_size']
        self.LEARNING_RATE = conf['learning_rate']

        self.CACHE_DIR = os.path.join(_base_dir, 'cache')

        # Create the CACHE_DIR if doesn't exist
        if not os.path.isdir(self.CACHE_DIR):
            os.mkdir(self.CACHE_DIR)

    @property
    def FEATURES_PATH(self):
        return os.path.join(self.CACHE_DIR, 'features.joblib')

    @property
    def MODEL_PATH(self):
        return os.path.join(self.CACHE_DIR, 'lstm_model.h5')

    @property
    def PLOT_PATH(self):
        return os.path.join(self.CACHE_DIR, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_history.png')
