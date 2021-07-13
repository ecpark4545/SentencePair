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
		text_word_embed = self.word_embedding(text)
		
		# data, batch_size, sorted_indices, unsorted_indices
		packed_word_embed = pack_padded_sequence(text_word_embed, text_length.cpu(), batch_first=True, enforce_sorted=False)
		text_seq_output, (last_hidden, _) = self.lstm(packed_word_embed)
		unpacked_text, _ = pad_packed_sequence(text_seq_output, batch_first=True)
		# 2, bs, dim -> bs, 2, dim
		last_hidden = last_hidden.permute(1,0,2).contiguous().view(-1,1,2*self.hidden_size)
		return mask, text_length, text_word_embed, unpacked_text, last_hidden
		
		
	def get_mask(self, tensor):
		# if [pad] then value (1)
		mask = (tensor != 0).float()
		return mask



class GatedTrans(nn.Module):
	"""
		original code is from https://github.com/yuleiniu/rva (CVPR, 2019)
		They used tanh and sigmoid, but we used tanh and LeakyReLU for non-linear transformation function
	"""
	def __init__(self, in_dim, out_dim):
		super(GatedTrans, self).__init__()
		self.embed_y = nn.Sequential(nn.Linear(in_dim,out_dim),nn.Tanh())
		self.embed_g = nn.Sequential(nn.Linear(in_dim,out_dim),nn.LeakyReLU())

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, x_in):
		x_y = self.embed_y(x_in)
		x_g = self.embed_g(x_in)
		x_out = x_y * x_g

		return x_out


class TopicAggregation(nn.Module):
	def __init__(self, hparams):
		super(TopicAggregation, self).__init__()
		self.hparams = hparams
		self.embedding_size = hparams.embedding_size
		self.hidden_size = hparams.hidden_size
		
		self.query_emb = nn.Sequential(
			nn.Dropout(p=hparams.drop_rate),
			GatedTrans(
				self.hidden_size * 2,
				self.hidden_size
			)
		)
		self.key_emb = nn.Sequential(
			nn.Dropout(p=hparams.drop_rate),
			GatedTrans(
				self.hidden_size * 2,
				self.hidden_size
			)
		)
		self.softmax = nn.Softmax(dim=-1)

		self.topic_gate = nn.Sequential(
			nn.Linear(self.embedding_size * 2, self.embedding_size * 2),
			nn.Sigmoid()
		)
		
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, query_word, query_feature, query_mask, key_word, key_feature, key_mask):
		bs, seq_len, bilstm = query_feature.size()
		
		# non-linear transformation
		query_feat = self.query_emb(query_feature)
		key_feat = self.key_emb(key_feature)

		# attention score / bs seq_len1 seq_len2
		qk_dot_score = torch.bmm(query_feat * query_mask.unsqueeze(-1), (key_feat * key_mask.unsqueeze(-1)).permute(0, 2, 1).contiguous())
		query_key_attn = self.masked_softmax(qk_dot_score, key_mask)
		attn_query = self.weighted_sum(query_key_attn, key_word, query_mask)
		
		# concat word emb & gate  / bs seq_len1 seq_len2
		word_add_attn_query = torch.cat((attn_query, query_word), dim=-1)
		topic_gate = self.topic_gate(word_add_attn_query)
		topic_aware_feat = topic_gate * word_add_attn_query
	
		return topic_aware_feat
	
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
	
	
class ContextMatching(nn.Module):
	def __init__(self, hparams):
		super(ContextMatching, self).__init__()
		self.hparams = hparams
		self.hidden_size = hparams.hidden_size

		# non-linear transformation
		self.query_emb = nn.Sequential(
			nn.Dropout(p=hparams.drop_rate),
			GatedTrans(self.hidden_size * 2, self.hidden_size))

		self.key_emb = nn.Sequential(
			nn.Dropout(p=hparams.drop_rate),
			GatedTrans(
				self.hidden_size * 2,
				self.hidden_size
			)
		)

		self.att = nn.Sequential(
			nn.Dropout(p=hparams.drop_rate),
			nn.Linear(self.hidden_size, 1),
		)

		self.softmax = nn.Softmax(dim=-1)

		self.context_gate = nn.Sequential(
			nn.Linear((self.hidden_size * 2) * 2,
								(self.hidden_size * 2) * 2),
			nn.Sigmoid()
		)
		
		self.projection = nn.Sequential(
			nn.Dropout(hparams.drop_rate),
			nn.Linear((self.hidden_size * 2)*2, self.hidden_size * 2 ),
			nn.ReLU())

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, query, key):
		query_feat = self.query_emb(query)
		key_feat = self.key_emb(key)

		att_score = self.att(query_feat * key_feat).squeeze(-1)  # element_wise multiplication -> attention
		att_score = self.softmax(att_score)
		attn_key = (key * att_score.unsqueeze(-1)).sum(1, keepdim=True)  # weighted sum : question-relevant dialog history

		key_add_query_feat = torch.cat((query, attn_key), dim=-1)
		context_gate = self.context_gate(key_add_query_feat)
		context_aware_feat = context_gate * key_add_query_feat
		context_aware_feat = self.projection(context_aware_feat)

		return context_aware_feat


class WordSentAtt(nn.Module):
	def __init__(self, hparams):
		super(WordSentAtt, self).__init__()
		self.hparams = hparams
		self.embedding_size = hparams.embedding_size
		self.hidden_size = hparams.hidden_size
		
		# projection
		self.query_emb = nn.Sequential(
			nn.Dropout(hparams.drop_rate),
			nn.Linear(self.hidden_size * 2, self.embedding_size * 2),
			nn.ReLU())
		
		self.softmax = nn.Softmax(dim=-1)
		
		
		
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
	
	def forward(self, query, key, key_mask):
		# bs 1 dim
		query_feat = self.query_emb(query)
		# bs 1 seq
		qk_dot_score = torch.bmm(query_feat, (key * key_mask.unsqueeze(-1)).permute(0, 2, 1).contiguous())
		query_key_attn = self.masked_softmax(qk_dot_score, key_mask)
		attn_query = self.self_weighted_sum(query_key_attn, key)
		
		return attn_query
	
	def masked_softmax(self, tensor, mask, transpose=False):
		if transpose:
			tensor = tensor.permute(0, 2, 1).contiguous()
		mask = mask.unsqueeze(1)
		score = tensor * mask
		mask_inf = (mask.float() - 1.0) * 1e13
		score = self.softmax(score + mask_inf)
		return score
	
	def self_weighted_sum(self, score, tensor):
		weighted_sum = score.bmm(tensor)
		return weighted_sum