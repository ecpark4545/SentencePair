import argparse
import os
from glob import glob
from datetime import datetime

import torch
import yaml
from easydict import EasyDict as edict

from src.utils.utils import init_logger
from train import Train
from pred import Pred


def train_model(cfg):
  save_dir = cfg.save_path
  save_dir += f"/{cfg.model}"
  logger = init_logger(save_dir)
  logger.info("Hyper-parameters: %s" % str(cfg))
  cfg.save_path = save_dir
  torch.set_num_threads(cfg.cpu_workers)
  model = Train(cfg)
  model.train()


def predict_model(cfg):
  save_dir = cfg.save_path
  if not cfg.previous_weights: # single
    previous_weight = save_dir + cfg.previous_weight
    model = Pred(cfg)
    model.run_eval(previous_weight)
    
  else: # ensemble
    print('ensemble')
    previous_weights = save_dir + cfg.previous_weights
    ensemble_weights = []
    for i in glob(previous_weights+'/*.pth'):
      ensemble_weights.append(i)
    
    model = Pred(cfg)
    model.run_ensemble(ensemble_weights)
    

parser = argparse.ArgumentParser(description="Sentence Similarity")
parser.add_argument("--cfg_file", dest="cfg_file", default="cfg/base.yml", type=str,
                    help="Path to a config file listing reader, model and solver parameters.")
# learning schedule
parser.add_argument("--gpu", dest='gpu', default="7", type=str, help="GPUs ")
parser.add_argument("--bs", dest='batch_size', default=32, type=int)
parser.add_argument("--hs", dest='hidden_size', default=128, type=int)
parser.add_argument("--es", dest='embedding_size', default=128, type=int)
parser.add_argument("--lr", dest='learning_rate', default=1e-3, type=float)
parser.add_argument("--dr", dest='drop_rate', default=0., type=float)
parser.add_argument('--model', type=str, default='base', help='(|base|esim|bimpm|mvan)')
parser.add_argument('--ops', dest='optimizer', default='adam', type=str, help='(|adam|adamW|)')
parser.add_argument('--loss_type',  default='mse', type=str, help='(|ce|mse|)')
parser.add_argument('--cls', dest='classification',action='store_true', help='|ce|mse|')
parser.add_argument('--pred', dest='predict',action='store_true', help='only predict')
parser.add_argument('--inference', dest="previous_weight" ,type=str, default='""', help='pretrained_weights')




if __name__ == '__main__':
  args = parser.parse_args()
  with open(args.cfg_file) as f:
    conf = yaml.safe_load(f)
  cfg = edict(conf)
  cfg.update(vars(args))
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
  cfg.gpu_ids = list(range(len(cfg.gpu.split(","))))
  # Train
  if not cfg.predict:
    print("=" * 80)
    print(f"train {args.model} model")
    print("=" * 80)
    train_model(cfg)
    
  # Only Prediction (Inference)
  else:
    print("=" * 80)
    print(f"Inference {args.model} model")
    print("=" * 80)
    predict_model(cfg)

    