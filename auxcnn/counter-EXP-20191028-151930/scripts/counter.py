import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from model import NetworkQuery as Network
from model import RetrievalModel
from datasets import CUB_200_2011, Stanford_Dog
from torch.utils.data import DataLoader
from utils import *
from torchstat import stat

parser = argparse.ArgumentParser("NAS")
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')#8
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--code_size', type=int, default=32, help='random seed')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--load_db', type=str, default='',help='location of saved database')
parser.add_argument('--load_query', type=str, default='',help='location of saved query model')


args = parser.parse_args()

args.save = 'counter-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  assert os.path.isfile(args.load_db)
  assert os.path.isfile(args.load_query)
  database = RetrievalModel(args=args)
  utils.load_database(database, args.load_db)
  logging.info("db param size = %fMB", utils.count_parameters_in_MB(database))
  print(stat(database,(3,224,224)))
  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, args.code_size, args.layers, args.auxiliary, genotype)
  utils.load(model,args.load_query)
  model.drop_path_prob = args.drop_path_prob
  logging.info("query param size = %fMB", utils.count_parameters_in_MB(model))
  print(stat(model,(3,224,224)))


  
if __name__ == '__main__':
  main() 
