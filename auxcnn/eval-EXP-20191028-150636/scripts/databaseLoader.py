import torch
import os
import numpy as np

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat

class dbData(Dataset):
  def  __init__(self,args):
    self.inputs=torch.load(args.inputs_location)
    self.msetarget=torch.load(args.mse_location)

  def  __len__(self):
    self.inputs.shape[0]

  def __getitem__(self,idx):
    return self.inputs[idx],self.msetarget[idx]