# Copied and modified from https://github.com/JunzheJosephZhu/see_hear_feel/blob/master/src/models/encoders.py
import torch
import torchaudio
import torch.nn as nn
import timm
from matplotlib import pyplot as plt
import librosa
import pickle

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from diffusion_policy.common.pytorch_util import replace_submodules




class ForceEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self,x):
        """
        B,T,H,D
        """
       

        x = self.model(x)
        return x


if __name__ == '__main__':
    pass