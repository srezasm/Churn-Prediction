import os
from datetime import datetime


class Config:
    def __init__(self):
        self.cache_dir = 'cache'

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

    @property
    def dataset_path(self):
        return os.path.join(self.cache_dir, 'processed_dataset.csv')

    @property
    def features_path(self):
        return os.path.join(self.cache_dir, 'features.joblib')

    @property
    def model_path(self):
        return os.path.join(self.cache_dir, 'lstm_model.h5')

    @property
    def plot_path(self):
        return os.path.join(self.cache_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_history.png')
