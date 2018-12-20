import os


class BaseConfig:
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'Data')
