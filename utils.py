import sys
import csv
import numpy as np
from collections import defaultdict


class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<$UNK>'
    self.add_word(self.unknown, count=0)

  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print('{} total words with {} uniques'.format(
        self.total_words, len(self.word_freq)))

  def encode(self, word):
    assert type(word) is str
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)


def scores(confmat):
  posPrec = confmat[1][1] / (confmat[1][1] + confmat[0][1])
  posReca = confmat[1][1] / (confmat[1][1] + confmat[1][0])
  posEff1 = (2 * posPrec * posReca) / (posPrec + posReca)
  negPrec = confmat[0][0] / (confmat[0][0] + confmat[1][0])
  negReca = confmat[0][0] / (confmat[0][0] + confmat[0][1])
  negEff1 = (2 * negPrec * negReca) / (negPrec + negReca)
  print('             precision    recall  f1-score')
  print('   positive       %.2f      %.2f      %.2f' %
        (posPrec, posReca, posEff1))
  print('   negative       %.2f      %.2f      %.2f' %
        (negPrec, negReca, negEff1))
  print('\n')
  print('avg / total       %.2f      %.2f      %.2f' %
        ((posPrec + negPrec)/2, (posReca + negReca)/2, (posEff1 + negEff1)/2))

# glove2dict from https://github.com/cgpotts/cs224u


def glove2dict(src_filename):
  """GloVe Reader.

  Parameters
  ----------
  src_filename : str
      Full path to the GloVe file to be processed.

  Returns
  -------
  dict
      Mapping words to their GloVe vectors.

  """
  with open(src_filename, encoding='utf8') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    return {line[0]: np.array(list(map(float, line[1:]))) for line in reader}


def vec2dict(src_filename, cut=-1):
  """vec Reader.

  Parameters
  ----------
  src_filename : str
      Full path to the vec file to be processed.

  Returns
  -------
  dict
      Mapping words to their vec vectors.

  """
  with open(src_filename, encoding='utf8') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    vecDict = {}
    for line in reader:
      if len(line) > 2:
        vecDict[line[0]] = np.array(list(map(float, line[1: -1])))
    return vecDict

def vec2dictAlt(src_filename, cut=-1):
  """vec Reader.

  Parameters
  ----------
  src_filename : str
      Full path to the vec file to be processed.

  Returns
  -------
  dict
      Mapping words to their vec vectors.

  """
  with open(src_filename, encoding='utf8') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    vecDict = {}
    for line in reader:
      if len(line) > 2:
        vecDict[line[0]] = np.array(list(map(float, line[1:])))
    return vecDict
