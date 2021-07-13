import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

# Manhattan LSTM model
class STSBaselineModel(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.vocab_size = hparams.vocab_size
		self.hidden_size = hparams.hidden_size
		self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
		self.encoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=1)
	
	def forward(self, text1: torch.Tensor, text2: torch.Tensor) -> torch.Tensor:
		# bs, seq_len
		text1_lengths = text1.gt(0).long().sum(-1)
		text2_lengths = text2.gt(0).long().sum(-1)
		
		# bs, seq_len, w_dim
		text1_word_embeds = self.word_embedding(text1)
		text2_word_embeds = self.word_embedding(text2)
		
		packed_text1 = pack_padded_sequence(text1_word_embeds, text1_lengths.cpu(), batch_first=True, enforce_sorted=False)
		packed_text2 = pack_padded_sequence(text2_word_embeds, text2_lengths.cpu(), batch_first=True, enforce_sorted=False)
		
		text1_seq_output, _ = self.encoder(packed_text1)
		text2_seq_output, _ = self.encoder(packed_text2)
		
		unpacked_text1, _ = pad_packed_sequence(text1_seq_output, batch_first=True)
		unpacked_text2, _ = pad_packed_sequence(text2_seq_output, batch_first=True)
		
		# bs, seq_len; from g function R^{ref} -> R
		text1_repre = unpacked_text1.sum(1) / text1_lengths.unsqueeze(1)
		text2_repre = unpacked_text2.sum(1) / text2_lengths.unsqueeze(1)
		
		distance = (text1_repre - text2_repre).norm(1, dim=-1)
		similarity = (distance * -1).exp()
		return similarity