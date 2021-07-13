import csv
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from src.utils.checkpointing import CheckpointManager, load_checkpoint
from src.models.esim_ce import ESIMCrossEntropy
from src.models.esim_mse import ESIMMse
from src.models.baseline_mse import STSBaselineModel
from src.models.bimpm_ce import BIMPMCrossEntropy
from src.models.bimpm_mse import BIMPMMse
from src.models.mvan_ce import MVANCrossEntropy
from src.models.mvan_mse import MVANMse
from torch.nn.utils.rnn import pad_sequence
from src.data.data import STSDatasetCE, STSDatasetMSE
from src.utils.utils import set_seed


class Pred(nn.Module):
	def __init__(self, cfg: dict):
		super(Pred, self).__init__()
		self.cfg = cfg
		self.csv_path = cfg.save_path + "/submission.csv"
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
	
	def _build_dataloader(self, dataset_path: str, classification=True):
		max_len = self.cfg.max_len
		test_data_path = self.cfg.data_path + '/test.csv'
		self.test_examples = self._load_data(test_data_path)
		
		if classification:
			self.test_dataset = STSDatasetCE(self.test_examples, max_len)
		else:
			self.test_dataset = STSDatasetMSE(self.test_examples, max_len)
		
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, drop_last=False, shuffle=False,
		                                   collate_fn=self._collate_fn)
	
		
		
		print("=" * 80)
		description = f"DATALOADER FINISHED:"
		print(description)
		self._logger.info(description)
	
	def _collate_fn(self, features):
		text1_batch, text2_batch, _ = list(zip(*features))
		text1_batch_tensor = pad_sequence(text1_batch, batch_first=True, padding_value=0)
		text2_batch_tensor = pad_sequence(text2_batch, batch_first=True, padding_value=0)
		return text1_batch_tensor, text2_batch_tensor
	
	def _build_model(self):
		print("=" * 80)
		print(f'Building model...{self.cfg.model}')
		print(f"GPU: {torch.cuda.is_available()}")
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
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
		description = f"BUILD MODEL FINISHED: {self.cfg.model}"
		print(description)
		self._logger.info(description)
	
	
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
	
	def run_eval(self, checkpoint):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._build_dataloader(self.cfg.data_path, classification=self.cfg.classification)
		self._build_model()
		print("=" * 30)
		print("3. EVAL START")
		print("=" * 30)
		model_state_dict, _ = load_checkpoint(checkpoint)
		print("evaluation model loading completes! ->", checkpoint)
		
		if isinstance(self.model, nn.DataParallel):
			self.model.module.load_state_dict(model_state_dict)
		else:
			self.model.load_state_dict(model_state_dict)

		test_preds = []
		with torch.no_grad():
			self.model.eval()
			for test_batch in tqdm(self.test_dataloader):
				text1, text2 = (tensor.to(self.device) for tensor in test_batch)
				pred = self.model(text1, text2)
				test_preds.extend(pred.gt(0.5).long().tolist())
		
		with open(self.csv_path, "w") as f:
			f.write("id,label\n")
			for features, pred in zip(self.test_examples, test_preds):
				f.write(f"{features[0]},{pred}\n")
				
		print("Inference done")
	
	def run_ensemble(self, checkpoints):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._build_dataloader(self.cfg.data_path, classification=self.cfg.classification)
		self._build_model()
		print("=" * 30)
		print("3. EVAL START")
		print("=" * 30)
		results = []
		for checkpoint in checkpoints:
			model_state_dict, _ = load_checkpoint(checkpoint)
			print("evaluation model loading completes! ->", checkpoint)
			
			if isinstance(self.model, nn.DataParallel):
				self.model.module.load_state_dict(model_state_dict)
			else:
				self.model.load_state_dict(model_state_dict)
			
			test_preds = []
			with torch.no_grad():
				self.model.eval()
				for test_batch in tqdm(self.test_dataloader):
					text1, text2 = (tensor.to(self.device) for tensor in test_batch)
					pred = self.model(text1, text2)
					test_preds.extend(pred.gt(0.5).long().tolist())
			
			results.append(test_preds)
		
		# ensemble
		final_result = []
		results = np.array(results)
		results = np.transpose(results)
		for result in results:
			final_result.append(np.bincount(result).argmax())
		
		with open(self.csv_path, "w") as f:
			f.write("id,label\n")
			for features, pred in zip(self.test_examples, final_result):
				f.write(f"{features[0]},{pred}\n")
		
		print("Ensemble done")