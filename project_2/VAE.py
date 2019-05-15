from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
from torch.distribution.normal import Normal


class SentenceVAE(nn.Module):
	def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, num_layers, padding_idx, device="cpu"):
		super(SentenceVAE, self).__init__()
		self.tanh = nn.Tanh()
		self.linear_h = nn.Linear(z_dim, hidden_dim)
		self.z_dim = z_dim
		self.normal = Normal(torch.zeros(z_dim), torch.eye(z_dim))

		self.embedding = nn.Embedding(vocab_len, vocab_dim, padding_idx)
		self.lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers)
		self.linear = nn.Linear(hidden_dim, vocab_len)
		self.softmax = nn.Softmax()

		self.to(device)

	def forward(self, x):
		z = self.normal.sample()
		h = self.tanh(self.linear_h(z))
		e = self.embedding(x)
		f = 
		out,h = self.lstm(x,h)
		out = self.linear(out)
		return out

	def predict(self, x, h=None):
		x = self.embedding(x)
		if h:
			x, h = self.lstm(x,h)
		else: 
			x, h = self.lstm(x)
		out = self.linear(x)

		return out, h



