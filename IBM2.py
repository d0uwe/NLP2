'''
IBM model 2 
Tim Smit, Tessa Wagenaar and Douwe van der Wal
'''

from collections import defaultdict, Counter
from itertools import product
from random import random

import matplotlib.pyplot as plt
import numpy as np
import dill
import numpy as np

from aer import read_naacl_alignments, AERSufficientStatistics

def read_data(filename):
    with open(filename, 'r') as f:
        e = f.readlines()

    return e

class IBM2: 
    def __init__(self, e_lines, f_lines):
        self.t = {}
        self._vocab = {}
        self._cpd = {}
        self._alignment_cpd = {}
        self.e = e_lines
        self.f = f_lines
        self._init_t()

    def _init_t(self):

        en_words = set()
        for e_sen in self.e:
            for e in e_sen.split():
                en_words.add(e) 

        fr_words = set()
        for f_sen in self.f:
            for f in f_sen.split():
                fr_words.add(f) 

        self.en_words, self.fr_words = en_words, fr_words
        self.t = defaultdict(lambda: defaultdict(lambda: 1/len(en_words)))

    def jump(self, aj, j, l, m):
        return aj - floor(j * (l/m))

    def logprob_sentence(self, e, f):
        e = ["NULL"] + e.split()
        f = f.split()
        m = len(f)
        l = len(e)
        pa = 1 / (1 + l)
        log_pfe = sum([np.log(pa * sum([self.t[f[j]][e[a]] for a in range(l)])) for j in range(m)])
        return log_pfe


    def logprob(self):
        logprob = 0
        for e,f in zip(self.e, self.f):
            log_pfe = self.logprob_sentence(e,f)
            logprob += log_pfe
        return logprob


    def evaluate_aer(self):
        path = 'data/validation/dev.wa.nonullalign'
        gold_sets = read_naacl_alignments(path)

        val_e = read_data('data/validation/dev.e')
        val_f = read_data('data/validation/dev.f')

        # print(gold_sets)
        predictions = []
        for e,f in zip(val_e, val_f):
            predictions.append(self.viterbi(e,f))

        metric = AERSufficientStatistics()

        for gold, pred in zip(gold_sets, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)

        return metric.aer()



    def EM(self, iterations):
        logprobs = []
        aers = []

        for iter in range(iterations):
            print("Currently running iteration: {0}".format(iter))
            counts = defaultdict(lambda: defaultdict(float))
            total = defaultdict(float)
            s_total = defaultdict(float)

            print("E-step")
            for e_sen, f_sen in zip(self.e, self.f):
                e_sen = ["NULL"] + e_sen.split()
                f_sen = f_sen.split()

                ef = product(e_sen, f_sen)
                for e, f in ef:
                    s_total[e] += self.t[f][e]
                
                ef = product(e_sen, f_sen)
                for e, f in ef:
                    counts[f][e] += self.t[f][e] / s_total[e]
                    total[f] += self.t[f][e] / s_total[e]

            
            print("M-step")
            for e_sen, f_sen in zip(self.e, self.f):
                e_sen = ["NULL"] + e_sen.split()
                f_sen = f_sen.split()
                ef = product(e_sen, f_sen)

                for e, f in ef:
                    self.t[f][e] = counts[f][e] / total[f]

            print("Evaluation")
            aers.append(self.evaluate_aer())
            logprobs.append(self.logprob())

        return logprobs, aers

    def viterbi(self, e_sen, f_sen):
        '''
        Input:
            e_sen:  an english sentence
            f_sen:  a french sentence

        Output:
            a:      an optimal alignment
        '''
        e_sen = ["NULL"] + e_sen.split()
        f_sen = f_sen.split()
        l = len(e_sen)
        m = len(f_sen)
        matrix = np.zeros((len(e_sen), len(f_sen)))
        for i, e_word in enumerate(e_sen):
            for j, f_word in enumerate(f_sen):
                matrix[i][j] = self.A[self.jump(i,j,l,m)] * self.t[f_word][e_word]

        alignment=[]
        num_cols = len(f_sen)
        for i in range(num_cols):
            col = matrix[:,i]
            e_word = np.argmax(col)
            alignment += [e_word]
        
        return set(zip(range(1,len(f_sen)+1), alignment))









		


# Change to true if model should be loaded from pickle
load_model = False

e = read_data("data/training/hansards.36.2.e")#[:1000]
f = read_data("data/training/hansards.36.2.f")#[:1000]
ef = list(set(zip(e,f)))
e, f = zip(*ef)

if load_model:
    ibm1 = dill.load(open("ibm1.p", 'rb'))
    logprobs = dill.load(open("logprobs_ibm1.p", 'rb'))
    aers = dill.load(open("aers_ibm1.p", 'rb'))
else:
    ibm1 = IBM1(e, f)
    logprobs, aers = ibm1.EM(5)
    dill.dump(ibm1, open("ibm1.p", 'wb'))
    dill.dump(logprobs, open("logprobs_ibm1.p", 'wb'))
    dill.dump(aers, open("aers_ibm1.p", 'wb'))

ibm1.viterbi(e[1], f[1])


plt.figure()
plt.plot(logprobs)
plt.savefig("train_logprobs.png")
plt.close()

plt.figure()
plt.plot(aers)
plt.savefig("train_aers.png")
plt.close()

