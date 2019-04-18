'''
IBM model 1 
Tim Smit, Tessa Wagenaar and Douwe van der Wal
'''

from collections import defaultdict, Counter
from itertools import product

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
		print(1/len(en_words))

		self.t = defaultdict(lambda: defaultdict(lambda: 1/len(en_words)))

	def logprob(self):



	def EM(self):
		for iter in range(iterations):
			print("Currently running iteration: {0}".format(iter))
			counts = defaultdict(lambda: defaultdict(float))
			total = defaultdict(float)
			s_total = defaultdict(float)

			for i, (e_sen, f_sen) in enumerate(zip(self.e, self.f)):
				ef = product(e_sen, f_sen)
				for e, f in ef:
					s_total[e] += self.t[e][f]

				for e, f in ef:
					counts[e][f] += self.t[e][f] / s_total[e]
					total[f] += self.t[e][f] / s_total[e]

			for e in self.en_words:
				for f in self.fr_words:
					self.t[e][f] = counts[e][f] / total[f]
		# for e,f in product(list(self.en_words), list(self.fr_words)):
		# 	self.t[e][f] = counts[e][f] / total[f]




		





e = read_data("data/training/hansards.36.2.e")
f = read_data("data/training/hansards.36.2.f")
ibm1 = IBM1(e, f)
ibm1.EM()
print(imb1.t['limits'])
