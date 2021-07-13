# BIMPM reference -> https://github.com/galsang/BIMPM-pytorch
import torch.nn as nn
from src.models.mvan_modules import *

class MVANCrossEntropy(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.hidden_size = hparams.hidden_size
		self.embedding_size = hparams.embedding_size
		self.dropout = nn.Dropout(hparams.drop_rate)
		self.softmax = nn.Softmax(dim=-1)
		
		self.pre_embed = InitEmbed(hparams)
		self.hypo_embed = InitEmbed(hparams)
		
		self.PH_topic = TopicAggregation(hparams)
		self.HP_topic = TopicAggregation(hparams)
		
		self.PH_context = ContextMatching(hparams)
		self.HP_context = ContextMatching(hparams)
		
		self.PH_wordsent = WordSentAtt(hparams)
		self.HP_wordsent = WordSentAtt(hparams)
		
		self.classification = nn.Sequential(
			self.dropout,
			nn.Linear(self.embedding_size * 4 + self.hidden_size * 4, self.hidden_size * 2),
			nn.ReLU(),
			nn.Linear(self.hidden_size * 2, 2),
		)
		
	def forward(self, pre: torch.Tensor, hypo: torch.Tensor) -> torch.Tensor:
		# bs, seq_len / bs / bs, seq_len, word_size / bs, seq_len dim*2 / bs 1 dim*2
		pre_mask, pre_len, pre_word, pre_feat, pre_last = self.pre_embed(pre)
		hypo_mask, hypo_len, hypo_word, hypo_feat, hypo_last = self.hypo_embed(hypo)
		
		pre_by_hypo_word = self.PH_topic(pre_word, pre_feat, pre_mask, hypo_word, hypo_feat, hypo_mask)
		hypo_by_pre_word = self.HP_topic(hypo_word, hypo_feat, hypo_mask, pre_word, pre_feat, pre_mask)
		
		pre_by_hypo_sent = self.PH_context(pre_last, hypo_last)
		hypo_by_pre_sent = self.HP_context(hypo_last, pre_last)
		
		pre_by_hypo_feat = self.PH_wordsent(hypo_by_pre_sent, pre_by_hypo_word, pre_mask)
		hypo_by_pre_feat = self.HP_wordsent(pre_by_hypo_sent, hypo_by_pre_word, hypo_mask)
		
		v = torch.cat((pre_by_hypo_feat, pre_by_hypo_sent,
		                          hypo_by_pre_sent, hypo_by_pre_feat), dim=-1)
		v =v.squeeze(1)
		
		logits = self.classification(v)
		probabilities = self.softmax(logits)
		
		return logits, probabilities
		
		
	
	



