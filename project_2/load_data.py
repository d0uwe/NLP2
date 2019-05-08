from itertools import product
from random import random

import matplotlib.pyplot as plt
import numpy as np
import numpy as np

def read_data(filename):
    with open(filename, 'r') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            sentence = []
            for item in line.split(" "):
                if item[0] != "(":
                    sentence.append(item.replace(")","").replace("\n",""))
            data.append(sentence)
    return data

print(read_data("23.auto.clean")[0:20])