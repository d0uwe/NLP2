from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
from random import randint

import torch
import torch.optim as optim
from load_data import LoadData
from RNNLM import RNNLanguageModel
# from torch.utils.data import DataLoader

# from dataset import TextDataset
# from model import TextGenerationModel

def acc(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    """
    _, y_pred = predictions.max(1)
    y_gold = targets

    accuracy = torch.tensor(y_pred == y_gold, dtype=torch.float).mean()

    return accuracy


def perplexity(predictions, targets):
    return 1


def train(config):

    T = config.temperature

    results = {"acc": [], "loss": [], "sentences": [], "ppl": []}

    # Initialize the device which to run the model on
    # device = torch.device(config.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # Initialize the dataset and data loader (note the +1)
    dataset = LoadData("TRAIN_DATA")

    padding_idx = dataset.get_id("PAD")
    vocab_len = dataset.vocab_len
    vocab_dim = config.input_dim
    # Initialize the model that we are going to use
    model = RNNLanguageModel(vocab_len, vocab_dim, config.hidden_dim, config.lstm_num_layers, padding_idx, device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)  # fixme
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme
    softmax = torch.nn.Softmax()

    # Loop over data!!! TODO
    for step in range(int(config.train_steps)):
    # for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        model.train()
        optimizer.zero_grad()

        # Create x and y!! TODO
        sen = dataset.next_batch(config.batch_size)
        x = torch.tensor(sen[:-1]).to(device)
        y = torch.tensor(sen[1:]).to(device)

        out = model(x)

        out = out.view(-1, out.shape[2])

        loss = criterion(out, y.view(-1))   # fixme
        loss.backward()
        accuracy = acc(out, y.view(-1))   # fixme
        ppl = perplexity(out, y.view(-1))

        results["acc"].append(accuracy.item())
        results["loss"].append(loss.item())

        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{0}] Train Step {1}/{2}, Batch Size = {3}, Examples/Sec = {4:.2f}, "
                  "Accuracy = {5:.2f}, Loss = {6:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            model.eval()
            h = None
            sentence = []
            c = torch.randint(0, vocab_len - 1, (1,1), dtype=torch.long).to(device)
            c = c.to(device)
            for i in range(config.sample_length - 1):
                sentence.append(c.squeeze())
                out, h = model.predict(c, h)
                c = torch.tensor([[out.argmax()]])

            sentence.append(c.squeeze())
            sentence = torch.tensor(sentence)
            s = dataset.convert_to_string(sentence.tolist())
            print(s)
            results["sentences"].append(s)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    torch.save(model, open("LSTM.pt", 'wb'))

    print("Printing sentences:")

    for s in results["sentences"]:
        print(s)

    return results


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_dim', type=int, default=100, help='Length of an input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    # Added arguments
    parser.add_argument('--temperature', type=float, default=1.0, help='What temperature to use in the sentence sampling')
    parser.add_argument('--sample_length', type=int, default=50, help='Length of the sampled sentences')

    config = parser.parse_args()

    # Train the model
    train(config)
