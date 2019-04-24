'''
IBM model 1 
Tim Smit, Tessa Wagenaar and Douwe van der Wal
'''

from collections import defaultdict, Counter
from itertools import product
import matplotlib.pyplot as plt
# import pickle
import dill

from aer import read_naacl_alignments, AERSufficientStatistics

def read_data(filename):
	with open(filename, 'r') as f:
		e = f.readlines()

	return e

class IBM1: 
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

	def logprob(self):
		logprob = 0
		for e,f in zip(self.e, self.f):
			e = ["NULL"] + e.split()
			f = f.split()
			m = len(f)
			l = len(e)
			pa = 1 / (1 + l)
			log_pfe = sum([np.log(pa * sum([self.t[e[a]][f[j]] for a in range(l)])) for j in range(m)])
			logprob += log_pfe
		return logprob


	def EM(self, iterations):
		logprobs = []
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
					s_total[e] += self.t[e][f]
				
				ef = product(e_sen, f_sen)
				for e, f in ef:
					counts[e][f] += self.t[e][f] / s_total[e]
					total[f] += self.t[e][f] / s_total[e]

			
			print("M-step")
			for e_sen, f_sen in zip(self.e, self.f):
				e_sen = ["NULL"] + e_sen.split()
				f_sen = f_sen.split()
				ef = product(e_sen, f_sen)

				for e, f in ef:
					self.t[e][f] = counts[e][f] / total[f]

			logprobs.append(self.logprob())

		return logprobs

			# for e in self.en_words:
			# 	for f in self.fr_words:
			# 		if total[f] == 0:
			# 			self.t[e][f] = 0.0
			# 		else:
			# 			self.t[e][f] = counts[e][f] / total[f]

	def viterbi(self, e_sen, f_sen):
		'''
		Input:
			e_sen: 	an english sentence
			f_sen: 	a french sentence

		Output:
			a: 		an optimal alignment
		'''
		x=1






		


# Change to true if model should be loaded from pickle
load_model = False

e = read_data("data/training/hansards.36.2.e")#[:1000]
f = read_data("data/training/hansards.36.2.f")#[:1000]
print(Counter(zip(e,f)).most_common(10))
ef = list(set(zip(e,f)))
e, f = zip(*ef)

if load_model:
	ibm1 = dill.load(open("ibm1.p", 'r'))
	logprobs = dill.load(open("logprobs_ibm1.p", 'r'))
else:
	ibm1 = IBM1(e, f)
	logprobs = ibm1.EM(1)
	dill.dump(ibm1, open("ibm1.p", 'w'))
	dill.dump(logprobs, open("logprobs_ibm1.p", 'w'))


plt.plot(logprobs)
plt.show()



