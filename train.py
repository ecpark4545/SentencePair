import csv
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from src.models.esim_ce import ESIMCrossEntropy
from src.models.esim_mse import ESIMMse
from src.models.baseline_mse import STSBaselineModel
from src.models.bimpm_ce import BIMPMCrossEntropy
from src.models.bimpm_mse import BIMPMMse
from src.models.mvan_ce import MVANCrossEntropy
from src.models.mvan_mse import MVANMse
# from setproctitle import setproctitle
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from src.utils.checkpointing import CheckpointManager
from src.data.data import STSDatasetCE, STSDatasetMSE
from src.utils.utils import set_seed


class Train(nn.Module):
  def __init__(self, cfg: dict):
    super(Train, self).__init__()
    self.cfg = cfg
    # self.data_format = Tuple[str, List[int], List[int], float]
    
    set_seed(self.cfg.seed_num)
    torch.set_num_threads(self.cfg.cpu_workers)
    title = "CHAD_%d_MODEL_%s" % (self.cfg.batch_size, self.cfg.model)
    setproctitle(title)
    self._logger = logging.getLogger(__name__)
    
  def _load_data(self, dataset_path: str):
    # session_id, text1_ids, text2_ids, label(default: -1)
      examples = []
      with open(dataset_path) as f:
        csv_reader = csv.DictReader(f)
        for example in csv_reader:
          text1_ids = [int(token_id) + 1 for token_id in example["sentence1"].split(" ")]
          text2_ids = [int(token_id) + 1 for token_id in example["sentence2"].split(" ")]
          label = float(example["label"]) if "label" in example else -1.0
          examples.append((example["id"], text1_ids, text2_ids, label))
      return examples
  
  def _build_dataloader(self, dataset_path:str, classification=True):
    TRAIN_DATA_SIZE = 40000
    max_len = self.cfg.max_len
    train_data_path = self.cfg.data_path + '/train.csv'
    trainable_examples = self._load_data(train_data_path)
    dev_split_index = int(TRAIN_DATA_SIZE * 0.8)
    train_examples = trainable_examples[:dev_split_index]
    dev_examples = trainable_examples[dev_split_index:]
    
    if classification:
      self.train_dataset = STSDatasetCE(train_examples, max_len)
      self.dev_dataset = STSDatasetCE(dev_examples, max_len)
    else:
      self.train_dataset = STSDatasetMSE(train_examples, max_len)
      self.dev_dataset = STSDatasetMSE(dev_examples, max_len)
    
    self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=True, collate_fn=self._collate_fn)
    self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=64, drop_last=False, shuffle=False,collate_fn=self._collate_fn)
    
    self.iterations = len(self.train_dataset) // self.cfg.batch_size
    self.total_iter = self.iterations * self.cfg.num_epoch
    self.cfg.total_iter = self.total_iter
    
    print("=" * 80)
    description = f"DATALOADER FINISHED:"
    print(description)
    self._logger.info(description)
  
  def _collate_fn(self, features):
    text1_batch, text2_batch, labels = list(zip(*features))
    text1_batch_tensor = pad_sequence(text1_batch, batch_first=True, padding_value=0)
    text2_batch_tensor = pad_sequence(text2_batch, batch_first=True, padding_value=0)
    return text1_batch_tensor, text2_batch_tensor, torch.stack(labels)
  
  def _build_model(self):
    print("=" * 80)
    print(f'Building model...{self.cfg.model}')
    print(f"GPU: {torch.cuda.is_available()}")
    if self.cfg.previous_weight == "":
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.start_epoch = 1
      
    if self.cfg.model == 'esim':
      if self.cfg.classification:
        self.model = ESIMCrossEntropy(self.cfg).to(self.device)
      else:
        self.model = ESIMMse(self.cfg).to(self.device)
        
    elif self.cfg.model == 'bimpm':
      if self.cfg.classification:
        self.model = BIMPMCrossEntropy(self.cfg).to(self.device)
      else:
        self.model = BIMPMMse(self.cfg).to(self.device)
        
    elif self.cfg.model == 'base':
      self.model = STSBaselineModel(self.cfg).to(self.device)
      
    elif self.cfg.model == 'mvan':
      if self.cfg.classification:
        self.model = MVANCrossEntropy(self.cfg).to(self.device)
      else:
        self.model = MVANMse(self.cfg).to(self.device)
      
      
    
    
    print("=" * 80)
    description= f"BUILD MODEL FINISHED: {self.cfg.model}"
    print(description)
    self._logger.info(description)
    
  
  def _setup_training(self):
    self.save_checkpoint = "%s/%s" % (self.cfg.save_path, "checkpoints")
    self.tensorboard = "%s/%s" % (self.cfg.save_path, "tensorboard")
    if not os.path.exists(self.tensorboard):
      os.makedirs(self.tensorboard)
    if not os.path.exists(self.save_checkpoint):
      os.makedirs(self.save_checkpoint)
    self.cfg.save_dirpath = self.save_checkpoint
    self.summary_writer = SummaryWriter(self.tensorboard)
    if self.cfg.optimizer == "adam":
      self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.cfg.learning_rate),
                                  betas=(self.cfg.beta1,self.cfg.beta2),
                                  weight_decay=self.cfg.wd)
      
    elif self.cfg.optimizer == "adamW":
      self.optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.learning_rate),
                                  betas=(self.cfg.beta1, self.cfg.beta2),
                                  weight_decay=self.cfg.wd)
    if self.cfg.lr_decay:
      self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                      milestones=[3,6], gamma=0.25, verbose=False)
    if self.cfg.classification:
      self.criterion = nn.CrossEntropyLoss()
    else:
      self.criterion = nn.MSELoss()

    self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_checkpoint, config=self.cfg)
    print("=" * 80)
    print("TRAINING SETUP FINISHED")
  
  def _metric_fn(self, predict, label):
    n = label.size()[0]
    loss = self.criterion(predict, label)
    
    if self.cfg.classification:
      _, out_classes = predict.max(dim=1)
      correct = (out_classes == label).sum() / n
      return correct.item(), loss
    else:
      correct = []
      correct.extend(predict.gt(0.5).float().eq(label).long().tolist())
      return correct, loss
    
  def train(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._build_dataloader(self.cfg.data_path, classification=self.cfg.classification)
    self._build_model()
    self._setup_training()
    print("=" * 80)
    
    start_time = datetime.now().strftime('%H:%M:%S')
    self._logger.info("Start train model at %s" % start_time)
    
    global_iteration_step = 0
    train_begin = datetime.utcnow()  # New
    accu_loss = 0
    accu_cnt = 0
    accu_acc = 0
    best_valid = 0.
    
    for epoch in range(self.start_epoch, self.cfg.num_epoch + 1):
      self.model.train()
      tqdm_batch_iterator = tqdm(self.train_dataloader)
      train_loss_stack = []
      train_acc_stack = []
      for batch_idx, batch in enumerate(tqdm_batch_iterator):
        text1, text2, label = (tensor.to(self.device) for tensor in batch)
        
        if self.cfg.classification:
          logits, probabilities = self.model(text1, text2)
        else:
          logits = self.model(text1, text2)
          
        batch_acc, batch_loss = self._metric_fn(logits, label)
        batch_loss.backward()
        accu_acc += np.mean(batch_acc)
        accu_loss += batch_loss.item()
        accu_cnt += 1
        
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        self.optimizer.step()
        if self.cfg.lr_decay:
          self.scheduler.step()
        self.optimizer.zero_grad()
        
        train_loss_stack.append(batch_loss.item())
        train_acc_stack.append(batch_acc)
        
        global_iteration_step += 1
        running_lr = self.optimizer.param_groups[0]['lr']
        description = "[{}][Epoch: {:3d}][Iter: {:6d}][Acc: {:6f}][Loss: {:6f}][lr: {:7f}]".format(
          datetime.utcnow() - train_begin,
          epoch,
          global_iteration_step, accu_acc / accu_cnt, accu_loss / accu_cnt, running_lr)
        tqdm_batch_iterator.set_description(description)
        
      # -------------------------------------------------------------------------
      #   ON EPOCH END  (checkpointing and validation)
      # -------------------------------------------------------------------------
      description = f"EPOCH:{epoch} Train ACC {np.mean(train_acc_stack):.4f} Train Loss {np.mean(train_loss_stack):.4f}"
      print(description)
      self._logger.info(description)
      self.checkpoint_manager.step(epoch)
      self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
      self._logger.info(self.previous_model_path)
      # -------------------------------------------------------------------------
      #   Validation
      # -------------------------------------------------------------------------
      
      self._logger.info("Evaluation after %d epoch" % epoch)
      self.model.eval()
      accu_dev_loss, accu_dev_acc = 0, 0
      val_accu_cnt = 0
      for dev_batch in self.dev_dataloader:
        with torch.no_grad():
          text1, text2, label = (tensor.to(self.device) for tensor in dev_batch)
          if self.cfg.classification:
            logits, probabilities = self.model(text1, text2)
          else:
            logits = self.model(text1, text2)
            
          dev_acc, dev_loss = self._metric_fn(logits, label)
          accu_dev_loss += (dev_loss.item())
          accu_dev_acc += (np.mean(dev_acc))
          val_accu_cnt += 1
      description = f"EPOCH:{epoch} DEV ACC {accu_dev_acc/ val_accu_cnt:.4f} DEV Loss {accu_dev_loss/ val_accu_cnt:.4f}"
      print(description)
      self._logger.info(description)