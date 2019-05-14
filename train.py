import Probs

from itertools import tee, islice
from utils import read_corpus

smoother = 'loglinear1'
lexicon = 'data/lexicon/words-100.txt'
corpus = read_corpus(path='data/txt', regex='*Yonge*.txt')

lm = Probs.LanguageModel()
lm.set_smoother(smoother)
lm.read_vectors(lexicon)

corpus, train_corpus = tee(corpus)
lm.train(train_corpus)
lm.save('general-100.model')