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
import numpy as np 


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def comp_recon_loss(out, target, mask):
    out_flat = out.view(-1, out.shape[-1])
    log_probs_flat = torch.nn.functional.log_softmax(out_flat, dim=1)
    target_size = target.shape
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.reshape(target_size) * mask.float()
    # losses = losses * mask.float()
    loss = (losses.sum() / mask.float().sum()) / out.shape[0]
    return loss


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
        self.hidden2mean = nn.Linear(2*hidden_dim, z_dim)
        self.hidden2logvar = nn.Linear(2*hidden_dim, z_dim)

        # Decoder
        self.z2hidden = nn.Linear(z_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_len, vocab_dim)
        self.rnn_decoder = nn.GRU(vocab_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear_out = nn.Linear(2*hidden_dim, vocab_len)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # Encoder
        lengths = (x != self.padding_idx).sum(1)
        lengths, sort_idx = lengths.sort(descending=True)
        x = x[sort_idx,:].to(device)
        e = self.embedding(x)
        e = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True)
        _, h = self.rnn_encoder(e)
        fnb1 = h.reshape(h.shape[1], -1)
        mu = self.hidden2mean(fnb1)
        logvar = self.hidden2logvar(fnb1)
        sigma = torch.exp(logvar * 0.5)

        kl_losses = []
        recon_losses = []
        for i in range(10): # Compute Loss multiple times. 
            # Create latent vector
            epsilon = torch.randn(self.z_dim).to(device)
            z = mu + sigma * epsilon

            # Decoder
            h = self.tanh(self.z2hidden(z)).reshape(1, -1, self.hidden_dim)
            out, h = self.rnn_decoder(e)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            sentence = self.linear_out(out)

            mask = (x != self.padding_idx).reshape(x.shape)

            kl_loss = -0.5 * (1 + sigma.log() - mu.pow(2) - sigma).sum() / x.shape[0] 
            # recon_loss = nn.functional.cross_entropy(y_hat, y, reduction='sum', ignore_index=self.padding_idx)
            recon_loss = comp_recon_loss(sentence, x, mask)

            kl_losses.append(kl_loss.item())
            recon_losses.append(recon_loss.item())

        return (kl_loss + recon_loss), kl_losses, recon_losses

        
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

    results = {"ELBO mean": [], "ELBO std": [], "KL mean": [], "KL std": [], "sentences": []}

    for step in range(int(config.epochs)):
        # Only for time measurement of step through network
        t1 = time.time()

        model.train()
        optimizer.zero_grad()

        # Create x and y
        sen = dataset.next_batch(config.batch_size)
        x = torch.tensor(sen).to(device)
        y = torch.tensor(sen[:,1:]).to(device)
        loss, kl_losses, recon_losses = model(x)

        elbos = np.array(kl_losses) + np.array(recon_losses)
        loss.backward()
        optimizer.step()

        results["ELBO mean"].append(np.mean(elbos))
        results["ELBO std"].append(np.sqrt(np.var(elbos)))
        results["KL mean"].append(np.mean(kl_losses))
        results["KL std"].append(np.sqrt(np.var(kl_losses)))

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

    em = np.array(results["ELBO mean"])
    ev = np.array(results["ELBO std"])
    km = np.array(results["KL mean"])
    kv = np.array(results["KL std"])
    

    plt.figure(figsize=(12, 6))
    plt.plot(results["ELBO mean"], label='ELBO')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig("SentenceVAE_ELBO.png")    
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(results["KL mean"], label='KL')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('KL')
    plt.tight_layout()
    plt.savefig("SentenceVAE_KL.png")    

    pickle.dump(results, open("VAE_results.p", 'wb'))
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
