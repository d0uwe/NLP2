from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import torch.nn as nn
import torch
from torch.distributions.normal import Normal
from load_data import LoadData


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=1):
        super(Encoder, self).__init__()
        # self.tanh = nn.Tanh()
        # self.linear_h = nn.Linear(z_dim, hidden_dim)
        self.z_dim = z_dim
        self.normal = Normal(torch.zeros(z_dim), torch.eye(z_dim))
        self.embedding = nn.Embedding(vocab_len, vocab_dim, padding_idx)
        self.forward_lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers)
        self.backward_lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers)
        self.linear_h = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, z_dim)
        self.linear_std = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()


    def forward(self, x):
        x_backward = x.flip(1)
        e = self.embedding(x)
        e_backward = self.embedding(x_backward)
        f, hf = self.forward_lstm(e)
        b, hb = self.backward_lstm(e_backward)
        fn = f[:,-1]
        b1 = b[:,-1]
        fnb1 = torch.cat((fn, b1), 1)
        h = self.linear_h(fnb1)
        mu = self.linear_mean(h)
        sigma = self.softplus(self.linear_std(h))

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=1):
        self.linear_h = nn.Linear(z_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_len, vocab_dim)

    def forward(self, z, x):
        h = self.tanh(self.linear_h(z))
        e = self.embedding(x)
        return x



class SentenceVAE(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=1):
        super(SentenceVAE, self).__init__()
        self.encoder = Encoder(vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=num_layers)
        self.decoder = Decoder(vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=num_layers)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        return mean
        
    def predict(self, x, h=None):
        return x




def main():
    # Initialize the dataset and data loader (note the +1)
    dataset = LoadData("TRAIN_DATA")

    padding_idx = dataset.get_id("PAD")
    vocab_len = dataset.vocab_len
    vocab_dim = config.input_dim
    
    model = SentenceVAE(vocab_len, vocab_dim, config.z_dim, config.hidden_dim, padding_idx)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    for step in range(int(config.epochs)):
        # Only for time measurement of step through network
        t1 = time.time()

        model.train()
        optimizer.zero_grad()

        # Create x and y
        sen = dataset.next_batch(config.batch_size)
        x = torch.tensor(sen).to(device)
        # y = torch.tensor(sen[:,1:]).to(device)

        model(x)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--z_dim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--input_dim', default=100, type=int, 
                        help="embedding dimension")
    parser.add_argument('--hidden_dim', default=128, type=int, 
                        help="hidden dimension LSTMs")
    parser.add_argument('--batch_size', default=64, type=int, 
                        help="batch size")

    config = parser.parse_args()

    main()
