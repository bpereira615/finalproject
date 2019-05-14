import sys
import numpy as np

import Probs

# model to use for generation
assert(len(sys.argv) > 1)
file = sys.argv[1]

# number of tokens to generate
n = 10 if len(sys.argv) < 3 else int(sys.argv[2])

# load a given model
lm = Probs.LanguageModel.load(file)
vocab = list(lm.vocab)

# determine starting word token
weight = [lm.tokens.get((Probs.BOS, z), 0) for z in lm.vocab]
total = sum(weight)
weight = list(map(lambda x: x/total, weight))
start = np.random.choice(vocab, 1, weight)[0]

# initialize vectors
OOL_vec = lm.vectors[Probs.OOL]
x_vec = lm.vectors.get(Probs.BOS, OOL_vec)
y_vec = lm.vectors.get(start, OOL_vec)

sentence = []
for _ in range(n):

	# copmute unweighted probabilities
	unweighted_prob = []
	for z in lm.vocab:
		z_vec = lm.vectors.get(z, OOL_vec)
		unweighted_prob.append(lm.u_score(x_vec, y_vec, z_vec, max_theta=0))

	# randomly sample
	z = np.random.choice(vocab, 1, unweighted_prob)[0]

	# end of sentence reached, do not keep generating
	if z == Probs.EOS: break

	# update context
	x_vec = y_vec
	y_vec = lm.vectors.get(z, OOL_vec)
	sentence.append(z)

print(sentence)