# BIMPM reference -> https://github.com/galsang/BIMPM-pytorch
import torch.nn as nn
from src.models.bimpm_modules import *
class BIMPMMse(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.hidden_size = hparams.hidden_size
		self.dropout = nn.Dropout(hparams.drop_rate)
		self.num_layer = hparams.num_perspective
		
		self.pre_embed = InitEmbed(hparams)
		self.hypo_embed = InitEmbed(hparams)
		self.full_match = FullMatch(hparams)
		self.max_full_match = MaxFullMatch(hparams)
		self.attentive_match = Attentive_Match()
		
		# Aggregation Layer
		self.p_aggr_lstm = nn.Sequential(self.dropout,
		                                 nn.LSTM(self.num_layer * 8,
		                                         self.hidden_size,
		                                         num_layers=1, batch_first=True,
		                                         bidirectional=True))
		self.h_aggr_lstm = nn.Sequential(self.dropout,
		                                 nn.LSTM(self.num_layer * 8,
		                                         self.hidden_size,
		                                         num_layers=1, batch_first=True,
		                                         bidirectional=True))
		# Matching layer
		for i in range(1, 9):
			setattr(self, f'mp_w{i}',
			        nn.Parameter(torch.rand(self.num_layer, self.hidden_size)))
			
		# Prediction layer
		self.classification = nn.Sequential(self.dropout,
		                                    nn.Linear(4 * self.hidden_size, self.hidden_size * 2),
		                                    nn.Tanh(), self.dropout, nn.Linear(self.hidden_size * 2, 1)
		                                    )
		
		self.softmax = nn.Softmax(dim=-1)
		self.init_weights()
		
	def forward(self, pre: torch.Tensor, hypo: torch.Tensor) -> torch.Tensor:
		# bs, seq_len / bs / bs, seq_len, 2*hidden = dim
		pre_mask, pre_len, pre_fw, pre_bw, pre_fw_last, pre_bw_last = self.pre_embed(pre)
		hypo_mask, hypo_len, hypo_fw, hypo_bw, hypo_fw_last, hypo_bw_last = self.hypo_embed(hypo)
		
		# 1 matching layer - full matching
		# bs seq_len1 l // bs seq_len2 l
		mp_pre_full_fw = self.full_match(pre_fw, pre_mask, hypo_fw_last, self.mp_w1)
		mp_pre_full_bw = self.full_match(pre_bw, pre_mask, hypo_bw_last, self.mp_w2)
		mp_hypo_full_fw = self.full_match(hypo_fw, hypo_mask, pre_fw_last, self.mp_w1)
		mp_hypo_full_bw = self.full_match(hypo_bw, hypo_mask, pre_bw_last, self.mp_w2)
		
		# 2 matching layer - maxpooling matching
		# bs seq_len1, seq_len2, l
		mp_max_fw = self.max_full_match(pre_fw, pre_mask, hypo_fw, hypo_mask, self.mp_w3)
		mp_max_bw = self.max_full_match(pre_bw, pre_mask, hypo_bw, hypo_mask, self.mp_w4)
		# bs seq_len1 l // bs seq_len2 l
		mp_pre_max_fw, _ = mp_max_fw.max(dim=2)
		mp_pre_max_bw, _ = mp_max_bw.max(dim=2)
		mp_hypo_max_fw, _ = mp_max_fw.max(dim=1)
		mp_hypo_max_bw, _ = mp_max_bw.max(dim=1)
		
		# 3 matching layer - attentive matching
		# bs seq_len1, seq_len2
		att_fw = self.attentive_match(pre_fw, pre_mask, hypo_fw, hypo_mask)
		att_bw = self.attentive_match(pre_bw, pre_mask, hypo_bw, hypo_mask)
		
		# bs seq_len1, seq_len2, dim
		att_pre_fw = pre_fw.unsqueeze(2) * att_fw.unsqueeze(3)
		att_pre_bw = pre_bw.unsqueeze(2) * att_bw.unsqueeze(3)
		att_hypo_fw = hypo_fw.unsqueeze(1) * att_fw.unsqueeze(3)
		att_hypo_bw = hypo_bw.unsqueeze(1) * att_bw.unsqueeze(3)
		
		# bs seq_len1 dim // bs seq_len2 dim
		att_mean_pre_fw = self.div_with_small_value(att_pre_fw.sum(dim=1),
		                                            att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1).contiguous())
		att_mean_pre_bw = self.div_with_small_value(att_pre_bw.sum(dim=1),
		                                            att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1).contiguous())
		att_mean_hypo_fw = self.div_with_small_value(att_hypo_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
		att_mean_hypo_bw = self.div_with_small_value(att_hypo_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))
		
		# bs seq_len1 l // bs seq_len2 l
		mp_pre_att_fw = self.full_match(pre_fw, pre_mask, att_mean_hypo_fw, self.mp_w5, attn=True)
		mp_pre_att_bw = self.full_match(pre_bw, pre_mask, att_mean_hypo_bw, self.mp_w6, attn=True)
		mp_hypo_att_fw = self.full_match(hypo_fw, hypo_mask, att_mean_pre_fw, self.mp_w5, attn=True)
		mp_hypo_att_bw = self.full_match(hypo_bw, hypo_mask, att_mean_pre_bw, self.mp_w6, attn=True)
		
		# 4 matching layer - max attentive matching
		# bs seq_len1 dim // bs seq_len2 dim
		att_max_pre_fw, _ = att_pre_fw.max(dim=1)
		att_max_pre_bw, _ = att_pre_bw.max(dim=1)
		att_max_hypo_fw, _ = att_hypo_fw.max(dim=2)
		att_max_hypo_bw, _ = att_hypo_bw.max(dim=2)
		
		mp_pre_att_max_fw = self.full_match(pre_fw, pre_mask, att_max_hypo_fw, self.mp_w7, attn=True)
		mp_pre_att_max_bw = self.full_match(pre_bw, pre_mask, att_max_hypo_bw, self.mp_w8, attn=True)
		mp_hypo_att_max_fw = self.full_match(hypo_fw, hypo_mask, att_max_pre_fw, self.mp_w7, attn=True)
		mp_hypo_att_max_bw = self.full_match(hypo_bw, hypo_mask, att_max_pre_bw, self.mp_w8, attn=True)
		
		# bs seq_len1 l*8
		mp_pre = torch.cat(
			[mp_pre_full_fw, mp_pre_max_fw, mp_pre_att_fw, mp_pre_att_max_fw,
			 mp_pre_full_bw, mp_pre_max_bw, mp_pre_att_bw, mp_pre_att_max_bw], dim=2)
		mp_hypo = torch.cat(
			[mp_hypo_full_fw, mp_hypo_max_fw, mp_hypo_att_fw, mp_hypo_att_max_fw,
			 mp_hypo_full_bw, mp_hypo_max_bw, mp_hypo_att_bw, mp_hypo_att_max_bw], dim=2)
		
		
		# Aggregation
		_, (aggr_pre_last, _) = self.p_aggr_lstm(mp_pre)
		_, (aggr_hypo_last, _) = self.h_aggr_lstm(mp_hypo)
		
		x = torch.cat(
			[aggr_pre_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2),
			 aggr_hypo_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)], dim=1)

		# prediction
		logits = self.classification(x)

		return logits
		
	
	def div_with_small_value(self, n, d, eps=1e-8):
		# too small values are replaced by 1e-8 to prevent it from exploding.
		d = d * (d > eps).float() + eps * (d <= eps).float()
		return n / d
	
	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.kaiming_normal_(module.weight.data)
				nn.init.constant_(module.bias.data, 0.0)
			
			elif isinstance(module, nn.LSTM):
				nn.init.kaiming_normal_(module.weight_ih_l0.data)
				nn.init.orthogonal_(module.weight_hh_l0.data)
				nn.init.constant_(module.bias_ih_l0.data, 0.0)
				nn.init.constant_(module.bias_hh_l0.data, 0.0)
				hidden_size = module.bias_hh_l0.data.shape[0] // 4
				module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0
				
				if (module.bidirectional):
					nn.init.kaiming_normal_(module.weight_ih_l0_reverse.data)
					nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
					nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
					nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
					module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
		# ----- Matching Layer -----
		for i in range(1, 9):
			w = getattr(self, f'mp_w{i}')
			nn.init.kaiming_normal_(w)




