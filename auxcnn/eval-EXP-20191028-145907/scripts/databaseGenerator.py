
import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import utils

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat

from model import RetrievalModel
from datasets import CUB_200_2011, Stanford_Dog
from torch.utils.data import DataLoader
from utils import *

parser = argparse.ArgumentParser("DBGENERATOR")
parser.add_argument('--load_db', type=str, default='',help='location of saved database')
parser.add_argument('--root', type=str, default='datasets',help='location of data')
parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',help='dataset name (CUB_200_2011 or Stanford_Dogs)')
parser.add_argument('--code_size', type=int, default=32, help='random seed')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')#96
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')

args = parser.parse_args()

args.save = 'generator-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(int(args.gpu))
  if args.dataset_name == 'CUB_200_2011':
    args.root = os.path.join(args.root, 'CUB_200_2011')
    dataset = CUB_200_2011
  elif params.dataset_name == 'Stanford_Dogs':
    args.root = os.path.join(args.root, 'Stanford_Dogs')
    dataset = Stanford_Dog
  else:
    logging.info('Dataset %s does not exsist.' % args.dataset_name)
  train_queue=DataLoader(dataset(args.root, if_train=True), batch_size=args.batch_size,
                                shuffle=True, num_workers=8)


  database = RetrievalModel(args=args)
  database.cuda()
  assert os.path.isfile(args.load_db)
  utils.load_database(database, args.load_db)
  database.eval()
  inputs=[]
  msetarget=[]
  with torch.no_grad():
    for step, (input, target) in enumerate(train_queue):
      input=input.cuda()
      output=database(input)
      msetarget.append(output.data.cpu())
      inputs.append(input.data.cpu())
  msetarget=torch.cat(msetarget)
  inputs=torch.cat(inputs)
  print(inputs.shape)
  print(msetarget.shape)
  torch.save(inputs,os.path.join(args.save, 'inputs.pt'))
  torch.save(msetarget,os.path.join(args.save, 'msetarget.pt'))


if __name__ == '__main__':
  main() 

