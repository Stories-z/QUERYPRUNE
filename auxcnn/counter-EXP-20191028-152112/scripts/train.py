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
from optim import get_optimizer

parser = argparse.ArgumentParser("NAS")
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--gpu', type=str, default='2', help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--root', type=str, default='../datasets',help='location of data')
parser.add_argument('--code_size', type=int, default=32, help='random seed')
parser.add_argument('--load_query', type=str, default='',help='location of saved query model')
parser.add_argument('--continue_train', type=bool_flag, default=False,help='decide to continue training')
parser.add_argument('--optimizer', type=str, default='adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001',help='choose which specific optimizer to use')
parser.add_argument('--load_db', type=str, default='',help='location of saved database')
parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',help='dataset name (CUB_200_2011 or Stanford_Dogs)')


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
  else:
    torch.cuda.set_device(gpus[0])

  # torch.cuda.set_device(args.gpu)
  # cudnn.benchmark = True
  torch.manual_seed(args.seed)
  # cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, args.code_size, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  if args.continue_train:
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
  train_queue=DataLoader(dataset(args.root, if_train=True), batch_size=args.batch_size,
                                shuffle=True, num_workers=8)

  database = RetrievalModel(args=args)
  database.cuda()
  if len(gpus)>1:
    print("True")
    database = nn.parallel.DataParallel(database, device_ids=gpus, output_device=gpus[0])
    database = database.module

  assert os.path.isfile(args.load_db)
  utils.load_database(database, args.load_db)
  database.eval()

  criterion = nn.MSELoss()
  criterion = criterion.cuda()

  optimizer = get_optimizer(model.parameters(), args.optimizer)

  if len(gpus)>1:
    optimizer = nn.DataParallel(optimizer, device_ids=gpus).module
  
  for epoch in range(args.epochs):
    logging.info('epoch %d', epoch)
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    train(train_queue, model, criterion, optimizer,database)
    utils.save(model, os.path.join(args.save, 'weights.pt'))
  

def train(train_queue, model, criterion, optimizer,database):
  model.train()
  database.eval()
  total_loss=0
  count=0
  for step, (input, target) in enumerate(train_queue):
    input=input.cuda()
    msetarget=database(input)
    msetarget.detach()
    logits=model(input)
    optimizer.zero_grad()
    loss=criterion(logits,msetarget)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    if step%args.report_freq==0:
      logging.info('step %d MSE : %f',step,loss.item())
    count+=1
    total_loss+=loss.item()

  logging.info('AVERAGE MSE : %f',total_loss/count)

if __name__ == '__main__':
  main() 
