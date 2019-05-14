import fnmatch
import os
import re
import Probs


from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize 


# aggregate works per author for statistics
def author_works(path):
    works = defaultdict(list)
    for _, _, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                author, title = file.strip().split('___')
                works[author].append(title)
    return works


# tokenize gutenberg text files matching regex
def read_corpus(path, regex):
    ps = PorterStemmer()
    for file in os.listdir(path):

        if not (
            fnmatch.fnmatch(file, 'Abraham Lincoln*') or \
            fnmatch.fnmatch(file, 'Charles Dickens*') or \
            fnmatch.fnmatch(file, '*Yonge*') or \
            fnmatch.fnmatch(file, 'Mark Twain*') 
        ): continue
        with open(os.path.join(path, file)) as f:
            for sent in sent_tokenize(f.read()):
                yield Probs.BOS
                yield from map(lambda x: ps.stem(x.lower()), filter(lambda x: any(c.isalpha() for c in x), word_tokenize(sent)))
                yield Probs.EOS


# load pretrained models into memory for clustering analysis
def read_models(path, regex):
    models = []
    for file in os.listdir(path):
        if not fnmatch.fnmatch(file, regex): continue
        lm = Probs.LanguageModel.load(os.path.join(path, file))
        models.append((lm, file))

    return models
