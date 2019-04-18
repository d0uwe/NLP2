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

	# def create_vocab(self):
	# 	e_lines, f_lines = self.e, self.f

	# 	vocab = defaultdict(lambda: defaultdict(int))
	# 	alignment = defaultdict(lambda: defaultdict(float))

	# 	for i, (e_sen, f_sen) in enumerate(zip(e_lines, f_lines)):
	# 		e_sen = e_sen.strip().split()
	# 		f_sen = f_sen.strip().split()
	# 		ef = product(range(len(e_sen)), range(len(f_sen)))
	# 		for idx_e, idx_f in ef:
	# 			alignments[idx_e][idx_f] 
	# 			vocab[e][f] += self.t[e][f]

	# 	self._vocab = vocab

	# 	cpd = defaultdict(lambda: defaultdict(float))
	# 	for e, f_dict in vocab.items():
	# 		total = sum(f_dict.values())
	# 		for f, count in f_dict.items():
	# 			cpd[e][f] = f/total

	# 	self._cpd = cpd


	# def create_alignments():


	def EM(self):
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

		for e,f in product(list(self.en_words), list(self.fr_words)):
			if total[f] == 0:
				self.t[e][f] = 0
			else:
				self.t[e][f] = counts[e][f] / total[f]




		





e = read_data("data/training/hansards.36.2.e")
f = read_data("data/training/hansards.36.2.f")
ibm1 = IBM1(e, f)
ibm1.EM()
print(ibm1.t['limits'])
# print(e[:10])