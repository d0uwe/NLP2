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
from pdb import set_trace
import matplotlib.pyplot as plt
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=1):
        super(Encoder, self).__init__()
        # self.tanh = nn.Tanh()
        # self.linear_h = nn.Linear(z_dim, hidden_dim)
        self.z_dim = z_dim
        self.normal = Normal(torch.zeros(z_dim), torch.eye(z_dim))
        self.embedding = nn.Embedding(vocab_len, vocab_dim, padding_idx)
        self.rnn = nn.GRU(vocab_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear_h = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, z_dim)
        self.linear_std = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()


    def forward(self, x):
        x_backward = x.flip(1)
        e = self.embedding(x)
        e_backward = self.embedding(x_backward)
        _, fnb1 = self.rnn(e)
        print(fnb1[0].shape)
        h = self.linear_h(fnb1)
        mu = self.linear_mean(h)
        logvar = self.linear_std(h)
        sigma = torch.exp(logvar * 0.5)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, num_layers=1):
        super(Decoder, self).__init__()
        self.linear_h = nn.Linear(z_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_len, vocab_dim)
        self.lstm = nn.LSTM(vocab_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_dim, vocab_len)
        self.softmax = nn.Softmax(dim=2)
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

    def sample(self, z, BOS_id, sample_length):
        h = self.tanh(self.linear_h(z)).reshape(1, -1, self.hidden_dim)
        c = torch.zeros_like(h)
        x = torch.tensor([[BOS_id]])
        sample = [BOS_id]
        for i in range(sample_length):
            e = self.embedding(x)
            out, (h,c) = self.lstm(e, (h,c))
            x = self.softmax(out).multinomial(1).reshape(1,1)
            sample.append(x.item())

        return sample




class SentenceVAE(nn.Module):
    def __init__(self, vocab_len, vocab_dim, z_dim, hidden_dim, padding_idx, bos_id, num_layers=1):
        super(SentenceVAE, self).__init__()
        self.z_dim = z_dim
        self.vocab_size = vocab_len
        self.padding_idx = padding_idx
        self.softmax = nn.Softmax()
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.bos_id = bos_id

        # Encoder
        self.embedding = nn.Embedding(vocab_len, vocab_dim, padding_idx)
        self.rnn_encoder = nn.GRU(vocab_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.linear_h = nn.Linear(2*hidden_dim, hidden_dim)
        self.hidden2mean = nn.Linear(2*hidden_dim, z_dim)
        self.hidden2logvar = nn.Linear(2*hidden_dim, z_dim)

        # Decoder
        self.z2hidden = nn.Linear(z_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_len, vocab_dim)
        self.rnn_decoder = nn.GRU(vocab_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear_out = nn.Linear(2*hidden_dim, vocab_len)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, y):
        # Encoder
        e = self.embedding(x)
        _, h = self.rnn_encoder(e)
        fnb1 = h.reshape(h.shape[1], -1)
        mu = self.hidden2mean(fnb1)
        logvar = self.hidden2logvar(fnb1)
        sigma = torch.exp(logvar * 0.5)

        # Create latent vector
        epsilon = torch.randn(self.z_dim).to(device)
        z = mu + sigma * epsilon

        # Decoder
        h = self.tanh(self.z2hidden(z)).reshape(1, -1, self.hidden_dim)
        out, h = self.rnn_decoder(e)
        sentence = self.linear_out(out)

        y_hat = sentence.view(-1, self.vocab_size)
        _,sen = sentence.max(2)

        # print("IN: ", dataset.convert_to_string(x[0,:].tolist()))
        # print("OUT:", dataset.convert_to_string(sen[0,:].tolist()))
        y = y.view(-1)

        kl_loss = -0.5 * (1 + sigma.log() - mu.pow(2) - sigma).sum()
        recon_loss = nn.functional.cross_entropy(y_hat, y, reduction='sum', ignore_index=self.padding_idx)

        return (kl_loss + recon_loss) / x.shape[0]

        
    def sample(self, n_samples, BOS_id, sample_length):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_sens = []

        sos = torch.tensor([[self.bos_id]]).to(device)

        for _ in range(n_samples):
            no_h_flag = True
            e = self.embedding(sos)
            z = torch.randn(self.z_dim).to(device)
            sentence = [sos.item()]
            for i in range(sample_length):
                if no_h_flag:
                    out, h = self.rnn_decoder(e)
                    h_flag = False
                else:
                    out, h = self.rnn_decoder(e, h)

                _, word = self.linear_out(out).max(2)
                word2 = nn.functional.softmax(self.linear_out(out), dim=2).squeeze()
                sentence.append(word2.multinomial(1).item())

            sampled_sens.append(sentence)
 
        return sampled_sens


dataset = LoadData("TRAIN_DATA")

def main():
    # Initialize the dataset and data loader (note the +1)
    # dataset = LoadData("TRAIN_DATA")

    padding_idx = dataset.get_id("PAD")
    bos_id = dataset.get_id("BOS")
    vocab_len = dataset.vocab_len
    vocab_dim = config.input_dim
    
    model = SentenceVAE(vocab_len, vocab_dim, config.z_dim, config.hidden_dim, padding_idx, bos_id)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    results = {"ELBO": [], "sentences": []}

    for step in range(int(config.epochs)):
        # Only for time measurement of step through network
        t1 = time.time()

        model.train()
        optimizer.zero_grad()

        # Create x and y
        sen = dataset.next_batch(config.batch_size)
        x = torch.tensor(sen[:,:-1]).to(device)
        y = torch.tensor(sen[:,1:]).to(device)
        loss = model(x, y)

        # sen = torch.tensor(sen[:,:-1]).to(device)
        # loss = model(sen)
        loss.backward()
        optimizer.step()

        results["ELBO"].append(loss.item())

        print(f"[Step {step}] train elbo: {loss}")

        # Sample from model
        if step % config.sample_every == 0:
            sentences = model.sample(5, dataset.get_id("BOS"), config.sample_length)
            for s in sentences:
                sen = dataset.convert_to_string(s)
                print(sen)
                results["sentences"].append(sen)

    # Print all generated sentences
    for s in results["sentences"]:
        print(s)

    plt.figure(figsize=(12, 6))
    plt.plot(results["ELBO"], label='ELBO')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig("SentenceVAE_ELBO.png")    
    pickle.dump(results, open("VAE_results.p", 'rb'))
    # torch.save(model, open("SentenceVAE.pt", 'wb'))
    

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(config).items():
    print(key + ' : ' + str(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1e6, type=int,
                        help='max number of epochs')
    parser.add_argument('--z_dim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--input_dim', default=100, type=int, 
                        help="embedding dimension")
    parser.add_argument('--hidden_dim', default=128, type=int, 
                        help="hidden dimension LSTMs")
    parser.add_argument('--batch_size', default=64, type=int, 
                        help="batch size")
    parser.add_argument('--sample_every', default=100, type=int, 
                        help="sample interval")
    parser.add_argument('--sample_length', default=100, type=int, 
                        help="sample size")

    config = parser.parse_args()
    print_flags()

    main()
