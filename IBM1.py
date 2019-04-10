'''
IBM model 1 
Tim Smit, Tessa Wagenaar and Douwe van der Wal
'''

from collections import defaultdict


def read_data(filename_e, filename_f):
	with open(filename_e, 'r') as fe:
		e = fe.readlines()

	with open(filename_f, 'r') as ff:
		f = ff.readlines()

	return e, f


e, f = read_data("data/training/hansards.36.2.e", "data/training/hansards.36.2.f")
print(e[:10])