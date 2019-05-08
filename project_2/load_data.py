from itertools import product
from random import random

import matplotlib.pyplot as plt
import numpy as np


DATATYPEDICT = {
"TRAIN_DATA" : "Data/02-21.10way.clean",
"VALIDATION_DATA" : "Data/22.auto.clean",
"TEST_DATA" : "Data/23.auto.clean"}


class LoadData():

    def __init__(self, trainingType):
        """
    `   Initializes LoadData object. 
        
        Args:
        trainingType :  "TRAIN_DATA" or "VALIDATION_DATA" or "TEST_DATA"
        batch_size : int of size batch
        """

        self.filename = DATATYPEDICT[trainingType]
        self.data = self.read_data()
        self.batch_location = 0
        self.vocab_size = 0
        self.data_size = len(self.data)
        self._word_to_id = {}
        self._id_to_word = {}
        self.create_word_id()
        


    def read_data(self):
        with open(self.filename, 'r') as f:
            data = []
            lines = f.readlines()
            for line in lines:
                sentence = []
                for item in line.split(" "):
                    if item[0] != "(":
                        sentence.append(item.replace(")","").replace("\n","").lower())
                data.append(sentence)
        return data
    

    def next_batch(self, batch_size):
        if batch_size + self.batch_location > self.data_size:
            self.batch_location = 0
        batch_data = self.data[self.batch_location:batch_size + self.batch_location]
        batch_data = self.to_semi_tensor(batch_data)
        self.batch_location = batch_size + self.batch_location
        return batch_data
    
    def to_tensor(self, batch_data):
        return [self.get_id(item) for sentence in batch_data for item in sentence]

    def create_word_id(self):
        all_data = self.data
        all_data = [item for sentence in all_data for item in sentence]
        all_data.append("BOS")
        all_data.append("EOS")
        all_data.append("UNK")
        all_data.append("PAD")
        all_data = sorted(list(set(all_data)))
        self.vocab_size = len(all_data)
        self._word_to_id = { wo:i for i,wo in enumerate(all_data) }
        self._id_to_word = { i:wo for i,wo in enumerate(all_data) }

    def get_id(self, word):
        return self._word_to_id[word]

    def get_word(self, id):
        return self._id_to_word[id]





    

data = LoadData("TRAIN_DATA")
for i in range(0,4):
    print("omg")
    print(data.next_batch(4))

