from os.path import realpath, join
from pathlib import Path

PROJECT_PATH = Path(realpath(__file__)).parent.parent.parent
DATA_PATH = join(PROJECT_PATH, 'data')
CHECKPOINTS_PATH = join(PROJECT_PATH, 'checkpoints')
MODELS_PATH = join(PROJECT_PATH, 'models')
LOGS_PATH = join(PROJECT_PATH, 'logs')
SCALARS_PATH  = join(LOGS_PATH, 'scalars')
VIDEOS_PATH = join(PROJECT_PATH, 'videos')
