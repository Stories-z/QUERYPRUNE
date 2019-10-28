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

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')#96
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')#8
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--load_db', type=str, default='',help='location of saved database')
parser.add_argument('--root', type=str, default='datasets',help='location of data')
parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',help='dataset name (CUB_200_2011 or Stanford_Dogs)')
parser.add_argument('--code_size', type=int, default=32, help='random seed')
parser.add_argument('--load_query', type=str, default='',help='location of saved query network')

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  np.random.seed(args.seed)

  gpus = [int(i) for i in args.gpu.split(',')]
  if len(gpus) == 1:
    torch.cuda.set_device(int(args.gpu))

  # torch.cuda.set_device(args.gpu)
  # cudnn.benchmark = True
  torch.manual_seed(args.seed)
  # cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, args.code_size, args.layers, args.auxiliary, genotype)
  model.drop_path_prob = args.drop_path_prob
  model = model.cuda()
  assert os.path.isfile(args.load_query)
  utils.load(model,args.load_query)

  if len(gpus)>1:
    print("True")
    model = nn.parallel.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = model.module

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  if args.dataset_name == 'CUB_200_2011':
    args.root = os.path.join(args.root, 'CUB_200_2011')
    dataset = CUB_200_2011
  elif params.dataset_name == 'Stanford_Dogs':
    args.root = os.path.join(args.root, 'Stanford_Dogs')
    dataset = Stanford_Dog
  else:
    logging.info('Dataset %s does not exsist.' % args.dataset_name)

  dataloaders = {
            'test': DataLoader(dataset(args.root, if_train=False), batch_size=args.batch_size,
                            shuffle=False, num_workers=8),
            'database': DataLoader(dataset(args.root, if_train=True, if_database=True), batch_size=args.batch_size,
                                shuffle=False, num_workers=8)
        }



  database = RetrievalModel(args=args)
  database.cuda()
  assert os.path.isfile(args.load_db)
  utils.load_database(database, args.load_db)

  model.eval()
  database.eval()

  test_binary, test_label = utils.cal_result(dataloaders['test'], model, args)
  train_binary, train_label = utils.cal_result(dataloaders['database'], database, args)

  
  mAP = utils.compute_mAP(train_binary, test_binary, train_label, test_label)
  logging.info('mAP: %f' % mAP)

    


if __name__ == '__main__':
  main() 
