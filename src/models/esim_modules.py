import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class InitEmbed(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_size, padding_idx=0)
		self.lstm = nn.LSTM(hparams.embedding_size, hparams.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
		self.dropout = nn.Dropout(p=hparams.drop_rate)
		
	def forward(self, text):
		# bs, seq_len
		mask = self.get_mask(text)
		text_length = text.gt(0).long().sum(-1)
		text_word_embed = self.dropout(self.word_embedding(text))
		
		# data, batch_size, sorted_indices, unsorted_indices
		packed_word_embed = pack_padded_sequence(text_word_embed, text_length.cpu(), batch_first=True, enforce_sorted=False)
		text_seq_output, _ = self.lstm(packed_word_embed)
		unpacked_text, _ = pad_packed_sequence(text_seq_output, batch_first=True)
		
		return mask, text_length, unpacked_text
		
		
	def get_mask(self, tensor):
		# if [pad] then value (1)
		mask = (tensor != 0).float()
		return mask


class LocalInference(nn.Module):
	def __init__(self):
		super().__init__()
		self.softmax = nn.Softmax(dim=-1)
	
	def forward(self, pre_feature, pre_mask, hypo_feature, hypo_mask):
		# bs, seq_len_1, seq_len_2
		similarity_matrix = pre_feature.bmm(hypo_feature.permute(0, 2, 1).contiguous())
		
		# attn weights
		pre_hypo_attn = self.masked_softmax(similarity_matrix, hypo_mask)
		hypo_pre_attn = self.masked_softmax(similarity_matrix, pre_mask, transpose=True)
		
		# Weighted sum
		attn_pre = self.weighted_sum(pre_hypo_attn, hypo_feature, pre_mask)
		attn_hypo = self.weighted_sum(hypo_pre_attn, pre_feature, hypo_mask)
		return attn_pre, attn_hypo
	
	def masked_softmax(self, tensor, mask, transpose=False):
		if transpose:
			tensor = tensor.permute(0, 2, 1).contiguous()
		mask = mask.unsqueeze(1)
		score = tensor * mask
		mask_inf = (mask.float() - 1.0) * 1e13
		score = self.softmax(score + mask_inf)
		return score
	
	def weighted_sum(self, score, tensor, mask):
		weighted_sum = score.bmm(tensor)
		return weighted_sum * mask.unsqueeze(-1)
	
class Composition(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.lstm = nn.LSTM(hparams.hidden_size, hparams.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
		
	def forward(self, feature, feature_mask, feature_len):
		# make composition feature
		packed_embed = pack_padded_sequence(feature, feature_len.cpu(), batch_first=True, enforce_sorted=False)
		seq_output, _ = self.lstm(packed_embed)
		v_feat, _ = pad_packed_sequence(seq_output, batch_first=True)
		feature_mask = feature_mask.unsqueeze(-1)
		
		# pooling
		v_avg = torch.sum(v_feat * feature_mask, dim=1) / feature_len.unsqueeze(-1)
		mask_inf = (feature_mask.float() - 1.0) * 1e13
		v_max, _ = (v_feat * feature_mask + mask_inf).max(dim=1)
		
		return v_avg, v_max
		
		
		