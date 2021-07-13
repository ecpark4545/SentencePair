# ESIM reference -> https://github.com/coetaur0/ESIM
import torch.nn as nn
from src.models.esim_modules import *

class ESIMCrossEntropy(nn.Module):
	def __init__(self, hparams):
		super().__init__()
		self.hidden_size = hparams.hidden_size
		self.dropout = nn.Dropout(hparams.drop_rate)
		
		self.pre_embed = InitEmbed(hparams)
		self.hypo_embed = InitEmbed(hparams)
		self.local_inference = LocalInference()
		
		self.projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
		                                nn.ReLU(), self.dropout)
		
		
		self.pre_comp = Composition(hparams)
		self.hypo_comp = Composition(hparams)
		self.softmax = nn.Softmax(dim=-1)
		
		self.classification = nn.Sequential(self.dropout,
		                                    nn.Linear(2 * 4 * self.hidden_size, self.hidden_size),
		                                    nn.Tanh(), self.dropout,
		                                    nn.Linear(self.hidden_size, 2)
		                                    )
		
		self.init_weights()
		
	def forward(self, pre: torch.Tensor, hypo: torch.Tensor) -> torch.Tensor:
		# bs, seq_len / bs / bs, seq_len, 2*hidden = dim
		pre_mask, pre_len, pre_feature = self.pre_embed(pre)
		hypo_mask, hypo_len, hypo_feature = self.hypo_embed(hypo)
		attn_pre, attn_hypo = self.local_inference(pre_feature, pre_mask, hypo_feature, hypo_mask)
		
		enhanced_pre = torch.cat([pre_feature, attn_pre,
		                          pre_feature - attn_pre,
		                          pre_feature * attn_pre],
		                         dim=-1)
		enhanced_hypo = torch.cat([hypo_feature, attn_hypo,
		                           hypo_feature - attn_hypo,
		                           hypo_feature * attn_hypo],
		                          dim=-1)
		
		proj_pre = self.projection(enhanced_pre)
		proj_hypo = self.projection(enhanced_hypo)
		
		v_a_avg, v_a_max = self.pre_comp(proj_pre, pre_mask, pre_len)
		v_b_avg, v_b_max = self.hypo_comp(proj_hypo, hypo_mask, hypo_len)
		v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
		logits = self.classification(v)
		probabilities = self.softmax(logits)
		return logits, probabilities
	
	def init_weights(self):
		"""
		Initialise the weights of the ESIM model.
		"""
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				nn.init.constant_(m.bias.data, 0.0)
			
			elif isinstance(m, nn.LSTM):
				nn.init.xavier_uniform_(m.weight_ih_l0.data)
				nn.init.orthogonal_(m.weight_hh_l0.data)
				nn.init.constant_(m.bias_ih_l0.data, 0.0)
				nn.init.constant_(m.bias_hh_l0.data, 0.0)
				hidden_size = m.bias_hh_l0.data.shape[0] // 4
				m.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0
				
				if (m.bidirectional):
					nn.init.xavier_uniform_(m.weight_ih_l0_reverse.data)
					nn.init.orthogonal_(m.weight_hh_l0_reverse.data)
					nn.init.constant_(m.bias_ih_l0_reverse.data, 0.0)
					nn.init.constant_(m.bias_hh_l0_reverse.data, 0.0)
					m.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0




