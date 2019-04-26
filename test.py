import dill
from IBM1 import IBM1, read_data
from IBM2 import IBM2
from collections import defaultdict

# filename = 

print("Making classes")

e = read_data("data/training/hansards.36.2.e")#[:1000]
f = read_data("data/training/hansards.36.2.f")#[:1000]

ibm1 = IBM1(e,f)
ibm2 = IBM2(e,f, 237)

print("loading pretrained classes")

ibm = dill.load(open("ibm2_expansion.p", 'rb'))

ibm2.t = ibm.t

test_e = read_data("data/testing/test/test.e")
test_f = read_data("data/testing/test/test.f")

predictions = []
for te, tf in zip(test_e, test_f):
	pred = ibm2.viterbi(te, tf)
	predictions.append(pred)

with open("ibm2.mle.naacl", 'w') as fl:
	for i, preds in enumerate(predictions):
		for alignment in preds:
			fl.write("{} {} {}".format(i, alignment[0], alignment[1]))

	fl.close()
