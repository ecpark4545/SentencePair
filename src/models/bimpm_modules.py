import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class InitEmbed(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.hidden_size = hparams.hidden_size
		self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_size, padding_idx=0)
		self.lstm = nn.LSTM(hparams.embedding_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
		self.dropout = nn.Dropout(p=hparams.drop_rate)
		
	def forward(self, text):
		# bs, seq_len
		mask = self.get_mask(text)
		text_length = text.gt(0).long().sum(-1)
		text_word_embed = self.dropout(self.word_embedding(text))
		
		# data, batch_size, sorted_indices, unsorted_indices
		packed_word_embed = pack_padded_sequence(text_word_embed, text_length.cpu(), batch_first=True, enforce_sorted=False)
		text_seq_output, (last_hidden, _) = self.lstm(packed_word_embed)
		unpacked_text, _ = pad_packed_sequence(text_seq_output, batch_first=True)
		fw, bw = torch.split(unpacked_text, self.hidden_size, dim=-1)
		
		return mask, text_length, fw, bw, last_hidden[-2,:,:], last_hidden[-1,:,:]
		
		
	def get_mask(self, tensor):
		# if [pad] then value (1)
		mask = (tensor != 0).float()
		return mask


class FullMatch(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.num_layer = hparams.num_perspective
		self.cos = nn.CosineSimilarity(dim=2)
		
	def forward(self, v1, v1_mask, v2, w, attn=False):
		# bs seq_len dim
		v1 = v1 * v1_mask.unsqueeze(dim=-1)
		if not attn:
			v2 = torch.stack([v2] * v1.size()[1], dim=1)
		
		# 1, 1, dim, l
		w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
		
		# bs seq_len dim l
		v1 = w * torch.stack([v1] * self.num_layer, dim=3)
		v2 = w * torch.stack([v2] * self.num_layer, dim=3)
		
		# bs seq_len dim l -> bs seq_len l
		m = self.cos(v1, v2)
		return m
	
class MaxFullMatch(nn.Module):
	def __init__(self,hparams):
		super().__init__()
		self.num_layer = hparams.num_perspective
		
	def div_with_small_value(self, n, d, eps=1e-8):
		# too small values are replaced by 1e-8 to prevent it from exploding.
		d = d * (d > eps).float() + eps * (d <= eps).float()
		return n / d
	
	def forward(self, v1, v1_mask, v2, v2_mask, w):
		# bs seq_len dim
		v1 = v1 * v1_mask.unsqueeze(dim=-1)
		v2 = v2 * v2_mask.unsqueeze(dim=-1)
		
		# 1, l, 1, dim
		w = w.unsqueeze(0).unsqueeze(2)
		
		# bs l, seq_len dim
		v1 = w * torch.stack([v1] * self.num_layer, dim=1)
		v2 = w * torch.stack([v2] * self.num_layer, dim=1)
		bs, l, v1_len, hidden = v1.size()
		_, _, v2_len, _ = v2.size()
		
		# bs, l, seq_len, 1
		v1_norm = v1.norm(p=2, dim=3, keepdim=True)
		v2_norm = v2.norm(p=2, dim=3, keepdim=True)
		
		# bs*l , seq_len, dim
		v1 = v1.reshape(bs * l, v1_len, hidden)
		v2 = v2.reshape(bs * l, v2_len, hidden)
		
		## manual cosine similarity
		# bs*l , seq_len, 1
		v1_norm = v1_norm.reshape(bs * l, v1_len, 1)
		v2_norm = v2_norm.reshape(bs * l, v2_len, 1)
		
		# bs*l v1_len, v2_len
		n = v1.bmm(v2.permute(0, 2, 1).contiguous())
		# bs*l v1_len, v2_len
		d = v1_norm * v2_norm.permute(0, 2, 1).contiguous()
		m = self.div_with_small_value(n, d).reshape(bs, v1_len, v2_len, l)
		
		return m
	
class Attentive_Match(nn.Module):
	def __init__(self):
		super().__init__()
	
	def div_with_small_value(self, n, d, eps=1e-8):
		# too small values are replaced by 1e-8 to prevent it from exploding.
		d = d * (d > eps).float() + eps * (d <= eps).float()
		return n / d
	
	def forward(self, v1, v1_mask, v2, v2_mask):
		# bs seq_len dim
		v1 = v1 * v1_mask.unsqueeze(dim=-1)
		v2 = v2 * v2_mask.unsqueeze(dim=-1)
		
		# bs seq_len 1
		v1_norm = v1.norm(p=2, dim=2, keepdim=True)
		v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1).contiguous()
		# bs v1_len v2_len
		a = v1.bmm(v2.permute(0, 2, 1).contiguous())
		d = v1_norm * v2_norm
		
		return self.div_with_small_value(a, d)