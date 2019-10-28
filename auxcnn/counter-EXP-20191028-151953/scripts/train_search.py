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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from model import RetrievalModel
from datasets import CUB_200_2011, Stanford_Dog
from torch.utils.data import DataLoader
from optim import get_optimizer

parser = argparse.ArgumentParser("NAS")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='3', help='gpu device id, split with ","')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--load_db', type=str, default='',help='location of saved database')
parser.add_argument('--root', type=str, default='../datasets',help='location of data')
parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',help='dataset name (CUB_200_2011 or Stanford_Dogs)')
parser.add_argument('--code_size', type=int, default=32, help='random seed')
parser.add_argument('--optimizer', type=str, default='adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001',help='choose which specific optimizer to use')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.MSELoss()
  criterion = criterion.cuda()

  model = Network(args.init_channels, args.code_size, args.layers, criterion)
  model = model.cuda()

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
    logger.info('Dataset %s does not exsist.' % args.dataset_name)
  train_data=dataset(args.root, if_train=True)
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))
  print('train data length: ',split)
  train_queue=DataLoader(train_data, batch_size=args.batch_size,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), num_workers=8)

  database = RetrievalModel(args=args)
  database.cuda()
  assert os.path.isfile(args.load_db)
  utils.load_database(database, args.load_db)
  database.eval()

  optimizer = get_optimizer(model.parameters(), args.optimizer)

  for epoch in range(args.epochs):
    logging.info('epoch %d', epoch)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train(train_queue, model, criterion, optimizer, database)

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
