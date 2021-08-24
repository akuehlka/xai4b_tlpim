import os
import random
import numpy as np
import torch, torchvision

from detectron2.engine import DefaultPredictor, DefaultTrainer

os.environ["FORCE_CUDA"]="0"
os.environ["CUDA_AVAILABLE_DEVICES"]="0"
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class MaskRCNN():
    # THIS WILL SAVE OUTPUT TO A TEMP DIRECTORY
    OUTPUT_DIR = "."

    TRAIN_IMGNAMES = []
    TRAIN_MSKNAMES = []
    DATASET01_IMGNAMES = []
    DATASET01_MSKNAMES = []
    WARSAW_IMGNAMES = []
    WARSAW_MSKNAMES = []

    trainds_dicts = None
    dataset01_dicts = None
    warsaw_dicts = None
    iris_metadata = None
    dataset01_metadata = None
    wwpostm_metadata = None

    cfg = None
