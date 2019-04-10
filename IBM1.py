'''
IBM model 1 
Tim Smit, Tessa Wagenaar and Douwe van der Wal
'''

from collections import defaultdict
from itertools import product

from aer import read_naacl_alignments, AERSufficientStatistics

def read_data(filename):
	with open(filename, 'r') as f:
		e = f.readlines()

	return e

class IBM1: 
	def __init__(self):
		self._vocab = {}
		self._cpd = {}

	def create_vocab(self, e_lines, f_lines):
		vocab = defaultdict(lambda: defaultdict(int))
		for e_sen, f_sen in zip(e_lines, f_lines):
			e_sen = e_sen.strip().split()
			f_sen = f_sen.strip().split()
			ef = product(e_sen, f_sen)
			for e,f in ef:
				vocab[e][f] += 1

		self._vocab = vocab

		cpd = defaultdict(lambda: defaultdict(float))
		for e, f_dict in vocab.items():
			total = sum(f_dict.values())
			for f, count in f_dict.items():
				cpd[e][f] = f/total

		self._cpd = cpd






e = read_data("data/training/hansards.36.2.e")
f = read_data("data/training/hansards.36.2.f")
print(e[:10])