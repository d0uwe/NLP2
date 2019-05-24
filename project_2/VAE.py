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


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=1):
        super(Encoder, self).__init__()
        # self.tanh = nn.Tanh()
        # self.linear_h = nn.Linear(z_dim, hidden_dim)
        self.z_dim = z_dim
        self.normal = Normal(torch.zeros(z_dim), torch.eye(z_dim))
        self.embedding = nn.Embedding(vocab_len, vocab_dim, padding_idx)
        self.forward_lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers, batch_first=True)
        self.backward_lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers, batch_first=True)
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
        super(Decoder, self).__init__()
        self.linear_h = nn.Linear(z_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_len, vocab_dim)
        self.lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_dim, vocab_len)
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim

    def forward(self, z, x):
        h = self.tanh(self.linear_h(z)).reshape(1, -1, self.hidden_dim)
        c = torch.zeros_like(h)
        e = self.embedding(x)
        sentence = []
        for i in range(e.shape[1]):
            emb = e[:,i:i+1,:].reshape(e.shape[0], 1, self.vocab_dim)
            out, (h,c) = self.lstm(emb, (h, c))
            sentence.append(out)
        out = torch.cat(sentence, 1)
        sen = self.linear_out(out)
        return sen



class SentenceVAE(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=1):
        super(SentenceVAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=num_layers)
        self.decoder = Decoder(vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=num_layers)
        criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn(self.z_dim).to(device)
        z = mu + sigma * epsilon
        sentence = self.decoder(z, x)

        kl_loss = -0.5 * (1 + sigma.log() - mu.pow(2) - sigma).sum()
        recon_loss = nn.functional.cross_entropy(immean, x, reduction='sum')

        return (kl_loss + recon_loss) / x.shape[0]

        
    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_sens = []

        for _ in range(n_samples):
            z = torch.normal(torch.zeros(self.z_dim), torch.ones(self.z_dim)).to(device)
            sentence = self.decoder(z)
            # CREATE SENTENCE
            sampled_sens.append(sentence)
 
        # rt = np.sqrt(n_samples)
        # sampled_ims = torch.stack(sampled_ims, 0).reshape(rt, rt, 1, 28, 28)
        return sampled_ims, im_means




def main():
    # Initialize the dataset and data loader (note the +1)
    dataset = LoadData("TRAIN_DATA")

    padding_idx = dataset.get_id("PAD")
    vocab_len = dataset.vocab_len
    vocab_dim = config.input_dim
    
    model = SentenceVAE(vocab_len, vocab_dim, config.z_dim, config.hidden_dim, padding_idx)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for step in range(int(config.epochs)):
        # Only for time measurement of step through network
        t1 = time.time()

        model.train()
        optimizer.zero_grad()

        # Create x and y
        sen = dataset.next_batch(config.batch_size)
        x = torch.tensor(sen).to(device)
        # y = torch.tensor(sen[:,1:]).to(device)

        loss = model(x)
        loss.backward()
        optimizer.step()






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
