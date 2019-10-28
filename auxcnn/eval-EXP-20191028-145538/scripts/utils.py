import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
import itertools
import argparse

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  # state_dict = torch.load(model_path)
  # # create new ordererDict that does not contain 'module'
  # from collections import OrderedDict
  # new_state_dict = OrderedDict()
  # for k, v in state_dict.items():
  #   namekey = k[7:] # remove 'module'
  #   new_state_dict[namekey] = v
  # # load params
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def combination(iterable, r):
    pool = list(iterable)
    n = len(pool)

    for indices in itertools.permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield list(pool[i] for i in indices)


def get_triplets(labels):
    labels = labels.cpu().data.numpy()
    triplets = []

    for label in set(labels):
        label_mask = (labels == label)
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        anchor_positives = list(combination(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))

def triplet_hashing_loss(image_embedding, image_labels, margin=1):
    
    triplets = get_triplets(image_labels)
    tanh=nn.Tanh()
    image_embedding=tanh(image_embedding)
    ap_distances = (image_embedding[triplets[:, 0]] - image_embedding[triplets[:, 1]]).pow(2).sum(1)
    an_distances = (image_embedding[triplets[:, 0]] - image_embedding[triplets[:, 2]]).pow(2).sum(1)

    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()

def cal_result(data_loarder, model, params):
    binary_code = []
    labels = []
    tanh=nn.Tanh()
    with torch.no_grad():
        for image, label in data_loarder:
            labels.append(label)
            output = tanh(model(image.cuda()))
            binary_code.append(output.data.cpu())

    return torch.sign(torch.cat(binary_code)), torch.cat(labels)

def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    Ns = Ns.type(torch.FloatTensor)

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct.type(torch.FloatTensor), dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    
    mAP = torch.mean(torch.Tensor(AP))
    return mAP

def load_database(model, save_path):
    '''
    load database
    '''
    model.load(save_path)

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    '''
    Parse boolean arguments from the command line.
    '''
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


