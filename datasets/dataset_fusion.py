import os
import yaml
import json
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from utils.data_preparation import process_data, exr_loader


class FusedDataset(Dataset):

    def __init__(self, data_dir, split='train', **kwargs):