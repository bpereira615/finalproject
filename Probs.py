# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)


import math
import random
import re
import sys
from itertools import tee

import numpy as np
from scipy.misc import logsumexp

BOS = 'BOS'   # special word type for context at Beginning Of Sequence
EOS = 'EOS'   # special word type for observed token at End Of Sequence
OOV = 'OOV'    # special word type for all Out-Of-Vocabulary words
OOL = 'OOL'    # special word type for all Out-Of-Lexicon words
DEFAULT_TRAINING_DIR = "/usr/local/data/cs465/hw-lm/All_Training/"
OOV_THRESHOLD = 3  # minimum number of occurrence for a word to be considered in-vocabulary


# TODO for TA: Maybe we should use inheritance instead of condition on the
# smoother (similar to the Java code).
class LanguageModel:
  def __init__(self):
    # The variables below all correspond to quantities discussed in the assignment.
    # For log-linear or Witten-Bell smoothing, you will need to define some 
    # additional global variables.
    self.smoother = None       # type of smoother we're using
    self.lambdap = None        # lambda or C parameter used by some smoothers

    # The word vector for w can be found at self.vectors[w].
    # You can check if a word is contained in the lexicon using
    #    if w in self.vectors:
    self.vectors = None    # loaded using read_vectors()

    self.vocab = None    # set of words included in the vocabulary
    self.vocab_size = None  # V: the total vocab size including OOV.

    self.tokens = None      # the c(...) function
    self.types_after = None # the T(...) function

    self.progress = 0        # for the progress bar

    self.bigrams = None
    self.trigrams = None
    
    # the two weight matrices X and Y used in log linear model
    # They are initialized in train() function and represented as two
    # dimensional lists.
    self.X, self.Y = None, None  

    # self.tokens[(x, y, z)] = # of times that xyz was observed during training.
    # self.tokens[(y, z)]    = # of times that yz was observed during training.
    # self.tokens[z]         = # of times that z was observed during training.
    # self.tokens[""]        = # of tokens observed during training.
    #
    # self.types_after[(x, y)]  = # of distinct word types that were
    #                             observed to follow xy during training.
    # self.types_after[y]       = # of distinct word types that were
    #                             observed to follow y during training.
    # self.types_after[""]      = # of distinct word types observed during training.

  def prob(self, x, y, z):
    """Computes a smoothed estimate of the trigram probability p(z | x,y)
    according to the language model.
    """

    if self.smoother == "UNIFORM":
      return float(1) / self.vocab_size
    elif self.smoother == "ADDL":
      if x not in self.vocab:
        x = OOV
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV
      return ((self.tokens.get((x, y, z), 0) + self.lambdap) /
        (self.tokens.get((x, y), 0) + self.lambdap * self.vocab_size))

      # Notice that summing the numerator over all values of typeZ
      # will give the denominator.  Therefore, summing up the quotient
      # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
      # as is required for any probability function.

    elif self.smoother == "BACKOFF_ADDL":

      if x not in self.vocab:
        x = OOV
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV

      p_z = ((self.tokens.get(z, 0) + self.lambdap) /
        (self.tokens[""] + self.lambdap * self.vocab_size))
      
      p_zy = ((self.tokens.get((y, z), 0) + self.lambdap * self.vocab_size * p_z) /
        (self.tokens.get(y, 0) + self.lambdap * self.vocab_size))

      p_zyx = ((self.tokens.get((x, y, z), 0) + self.lambdap * self.vocab_size * p_zy) /
        (self.tokens.get((x, y), 0) + self.lambdap * self.vocab_size))
        
      return p_zyx


    elif self.smoother == "BACKOFF_WB":
      sys.exit("BACKOFF_WB is not implemented yet (that's your job!)")
    elif self.smoother == "LOGLINEAR":

      if x not in self.vocab:
        x = OOV
      if y not in self.vocab:
        y = OOV
      if z not in self.vocab:
        z = OOV

      OOL = self.vectors['OOL']
      x_vec, y_vec, z_vec = self.vectors.get(x, OOL), self.vectors.get(y, OOL), self.vectors.get(z, OOL)
      
      denom, max_theta = self.Z(x_vec, y_vec)
      num = self.u_score(x_vec, y_vec, z_vec, max_theta)

      return num[0,0]/denom
    else:
      sys.exit("%s has some weird value" % self.smoother)



  def u_score(self, x, y, z, max_theta):
    xz = np.dot(np.dot(x, self.X), z.T)
    yz = np.dot(np.dot(y, self.Y), z.T)

    return np.exp(np.subtract(np.add(xz, yz), max_theta))


  def Z(self, x, y):
    xz = np.dot(np.dot(x, self.X), self.E)
    yz = np.dot(np.dot(y, self.Y), self.E)
    add = np.add(xz, yz)
    max_theta = np.max(add)

    return np.sum(np.exp(np.subtract(add, max_theta))), max_theta

  def filelogprob(self, corpus):
    """Compute the log probability of the sequence of tokens in file.
    NOTE: we use natural log for our internal computation.  You will want to
    divide this number by log(2) when reporting log probabilities.
    """
    logprob = 0.0
    x, y = BOS, BOS
    for z in corpus:
      prob = self.prob(x, y, z)
      logprob += math.log(prob)
      x = y
      y = z
    return logprob

  def read_vectors(self, filename):
    """Read word vectors from an external file.  The vectors are saved as
    arrays in a dictionary self.vectors.
    """
    with open(filename) as infile:
      header = infile.readline()
      self.dim = int(header.split()[-1])
      self.vectors = {}
      for line in infile:
        arr = line.split()
        word = arr.pop(0)
        self.vectors[word] = np.matrix([float(x) for x in arr], dtype=np.float64)

  def train (self, corpus):
    """Read the training corpus and collect any information that will be needed
    by the prob function later on.  Tokens are whitespace-delimited.

    Note: In a real system, you wouldn't do this work every time you ran the
    testing program. You'd do it only once and save the trained model to disk
    in some format.
    """

    # Clear out any previous training
    self.tokens = { }
    self.types_after = { }
    self.bigrams = []
    self.trigrams = [];

    corpus, vocab_corpus = tee(corpus)

    # While training, we'll keep track of all the trigram and bigram types
    # we observe.  You'll need these lists only for Witten-Bell backoff.
    # The real work:
    # accumulate the type and token counts into the global hash tables.

    # If vocab size has not been set, build the vocabulary from training corpus
    if self.vocab_size is None:
      self.set_vocab_size(vocab_corpus)

    # We save the corpus in memory to a list tokens_list.  Notice that we
    # appended two BOS at the front of the list and a EOS at the end.  You
    # will need to add more BOS tokens if you want to use a longer context than
    # trigram.
    x, y = BOS, BOS  # Previous two words.  Initialized as "beginning of sequence"
    # count the BOS context
    self.tokens[(x, y)] = 1
    self.tokens[y] = 1

    tokens_list = [x, y]  # the corpus saved as a list
    # corpus = self.open_corpus(filename)
    for z in corpus:
      # substitute out-of-vocabulary words with OOV symbol
      if z not in self.vocab:
        z = OOV
      # substitute out-of-lexicon words with OOL symbol (only for log-linear models)
      if self.smoother == 'LOGLINEAR' and z not in self.vectors:
        z = OOL
      self.count(x, y, z)
      x=y; y=z
      tokens_list.append(z)
  
    if self.smoother == 'LOGLINEAR': 
      # Train the log-linear model using SGD.

      # Initialize parameters
      self.X = np.matrix([[0.0 for _ in range(self.dim)] for _ in range(self.dim)], dtype=np.float64)
      self.Y = np.matrix([[0.0 for _ in range(self.dim)] for _ in range(self.dim)], dtype=np.float64)


      OOL_vec = self.vectors['OOL']
      
      # create d x V matrix for easy Z calculation
      self.E = np.zeros((self.dim, len(self.vocab)))
      for idx, key in enumerate(self.vocab):
        self.E[:, idx] = self.vectors.get(key, OOL_vec)


      # Optimization parameters
      gamma0 = 0.5  # initial learning rate, used to compute actual learning rate
      epochs = 1  # number of passes

      self.N = len(tokens_list) - 2  # number of training instances

      # ******** COMMENT *********
      # In log-linear model, you will have to do some additional computation at
      # this point.  You can enumerate over all training trigrams as following.
      #
      # for i in range(2, len(tokens_list)):
      #   x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
      #
      # Note1: self.lambdap is the regularizer constant C
      # Note2: You can use self.show_progress() to log progress.
      #
      # **************************

      sys.stderr.write("Start optimizing.\n")


      tokens_list = tokens_list[:5]


      gamma = gamma0 
      diff = np.zeros((self.dim, self.dim))
      sys.stderr.write("Getting word embeddings.\n")
      _z_s = [(_z, self.vectors.get(_z, OOL_vec)) for _z in self.vocab]
      # iterate over epochs
      for epoch in range(epochs):

        sys.stderr.write("Calculating F score.\n")
        F = 0
        # calculate F score
        for idx in range(2, len(tokens_list)):
          x, y, z = tokens_list[idx - 2], tokens_list[idx - 1], tokens_list[idx]
          F += np.log(self.prob(x, y, z))
        F /= self.N

        F -= self.lambdap/self.N * np.sum(np.add(np.square(self.X), np.square(self.Y)))
        print('epoch {}: F={}'.format(epoch+1, F))


        # update gamma by halving every fifth epoch
        if epoch % 5 == 0: gamma /= 1.5
        for idx in range(2, len(tokens_list)):

          if idx % 100:
            print(100. * idx / self.N)

          x, y, z = tokens_list[idx - 2], tokens_list[idx - 1], tokens_list[idx]

          x_vec, y_vec, z_vec = self.vectors.get(x, OOL_vec), self.vectors.get(y, OOL_vec), self.vectors.get(z, OOL_vec)

          observe_x = np.dot(x_vec.T, z_vec)
          observe_y = np.dot(y_vec.T, z_vec)


          expect_x = np.zeros(observe_x.shape)
          expect_y = np.zeros(observe_y.shape)
          for _z, _z_vec in _z_s:

            prob = self.prob(x, y, _z)

            expect_x = np.add(expect_x, np.multiply(prob, np.dot(x_vec.T, _z_vec)))
            expect_y = np.add(expect_y, np.multiply(prob, np.dot(y_vec.T, _z_vec)))

          regu_x = np.multiply(2.0*self.lambdap/self.N, self.X) 
          regu_y = np.multiply(2.0*self.lambdap/self.N, self.Y)

          grad_X = np.subtract(np.subtract(observe_x, expect_x), regu_x)
          grad_Y = np.subtract(np.subtract(observe_y, expect_y), regu_y)


          self.X = np.add(self.X, np.multiply(gamma, grad_X))
          self.Y = np.add(self.Y, np.multiply(gamma, grad_Y))
    

    sys.stderr.write("Finished training on %d tokens\n" % self.tokens[""])

  def count(self, x, y, z):
    """Count the n-grams.  In the perl version, this was an inner function.
    For now, I am just using a class variable to store the found tri-
    and bi- grams.
    """
    self.tokens[(x, y, z)] = self.tokens.get((x, y, z), 0) + 1
    if self.tokens[(x, y, z)] == 1:       # first time we've seen trigram xyz
      self.trigrams.append((x, y, z))
      self.types_after[(x, y)] = self.types_after.get((x, y), 0) + 1

    self.tokens[(y, z)] = self.tokens.get((y, z), 0) + 1
    if self.tokens[(y, z)] == 1:        # first time we've seen bigram yz
      self.bigrams.append((y, z))
      self.types_after[y] = self.types_after.get(y, 0) + 1

    self.tokens[z] = self.tokens.get(z, 0) + 1
    if self.tokens[z] == 1:         # first time we've seen unigram z
      self.types_after[''] = self.types_after.get('', 0) + 1
    #  self.vocab_size += 1

    self.tokens[''] = self.tokens.get('', 0) + 1  # the zero-gram


  def set_vocab_size(self, corpus):
    """When you do text categorization, call this function on the two
    corpora in order to set the global vocab_size to the size
    of the single common vocabulary.

    NOTE: This function is not useful for the loglinear model, since we have
    a given lexicon.
     """
    count = {} # count of each word

    for z in corpus:
      count[z] = count.get(z, 0) + 1
      self.show_progress();

    self.vocab = set(w for w in count if count[w] >= OOV_THRESHOLD)

    self.vocab.add(OOV)  # add OOV to vocab
    self.vocab.add(EOS)  # add EOS to vocab (but not BOS, which is never a possible outcome but only a context)
    sys.stderr.write('\n')    # done printing progress dots "...."

    if self.vocab_size is not None:
      sys.stderr.write("Warning: vocab_size already set; set_vocab_size changing it\n")
    self.vocab_size = len(self.vocab)
    sys.stderr.write("Vocabulary size is %d types including OOV and EOS\n"
                      % self.vocab_size)

  def set_smoother(self, arg):
    """Sets smoother type and lambda from a string passed in by the user on the
    command line.
    """
    r = re.compile('^(.*?)-?([0-9.]*)$')
    m = r.match(arg)
    
    if not m.lastindex:
      sys.exit("Smoother regular expression failed for %s" % arg)
    else:
      smoother_name = m.group(1)
      if m.lastindex >= 2 and len(m.group(2)):
        lambda_arg = m.group(2)
        self.lambdap = float(lambda_arg)
      else:
        self.lambdap = None

    if smoother_name.lower() == 'uniform':
      self.smoother = "UNIFORM"
    elif smoother_name.lower() == 'add':
      self.smoother = "ADDL"
    elif smoother_name.lower() == 'backoff_add':
      self.smoother = "BACKOFF_ADDL"
    elif smoother_name.lower() == 'backoff_wb':
      self.smoother = "BACKOFF_WB"
    elif smoother_name.lower() == 'loglinear':
      self.smoother = "LOGLINEAR"
    else:
      sys.exit("Don't recognize smoother name '%s'" % smoother_name)
    
    if self.lambdap is None and self.smoother.find('ADDL') != -1:
      sys.exit('You must include a non-negative lambda value in smoother name "%s"' % arg)

  def open_corpus(self, filename):
    yield from read(filename)

  def num_tokens(self, filename):
    corpus = self.open_corpus(filename)
    num_tokens = sum([len(l.split()) for l in corpus]) + 1

    return num_tokens

  def show_progress(self, freq=5000):
    """Print a dot to stderr every 5000 calls (frequency can be changed)."""
    self.progress += 1
    if self.progress % freq == 1:
      sys.stderr.write('.')

  @classmethod
  def load(cls, fname):
    import pickle
    fh = open(fname, mode='rb')
    loaded = pickle.load(fh)
    fh.close()
    return loaded

  def save(self, fname):
    import pickle
    with open(fname, mode='wb') as fh:
      pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
