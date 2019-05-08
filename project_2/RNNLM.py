from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class RNNLanguageModel(nn.Module):
	def __init__(self, vocab_len, vocab_dim, hidden_dim, num_layers, padding_idx, device="cpu"):
		super(RNNLanguageModel, self).__init__()
		self.h0 = torch.zeros(hidden_dim)
		self.embedding = nn.Embedding(vocab_len, vocab_dim, padding_idx)
		self.lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers)
		self.linear = nn.Linear(hidden_dim, vocab_len)
		self.softmax = nn.Softmax()

		self.to(device)

	def forward(self, x):
		h = self.h0 
		x = self.embedding(x)
		out,h = self.lstm(x)
		out = self.linear(out)
		return out

	def predict(self, x, h=None):
		return x
