from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import List, Dict, Union, Tuple
import torch
class STSDatasetCE(Dataset):
	def __init__(self, datas, max_len):
		self.datas = datas
		self.max_len = max_len
	
	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		_, text1, text2, label = self.datas[index]
		len_text1, len_text2 = len(text1), len(text2)
		if len_text1 > 20:
			text1 = text1[:20]
		if len_text2 > 20:
			text2 = text2[:20]
		return torch.tensor(text1), torch.tensor(text2), torch.tensor(label).long()
	
	def __len__(self) -> int:
		return len(self.datas)


class STSDatasetMSE(Dataset):
	def __init__(self, datas, max_len):
		self.datas = datas
		self.max_len = max_len
	
	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		_, text1, text2, label = self.datas[index]
		len_text1, len_text2 = len(text1), len(text2)
		if len_text1 > 20:
			text1 = text1[:20]
		if len_text2 > 20:
			text2 = text2[:20]
		return torch.tensor(text1), torch.tensor(text2), torch.tensor(label).float()
	
	def __len__(self) -> int:
		return len(self.datas)
