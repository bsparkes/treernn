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
  print('positive precision: %f' % posPrec)
  print('positive recall: %f' % posReca)
  print('positive f1: %f' % posEff1)
  print('negative precision: %f' % negPrec)
  print('negative recall: %f' % negReca)
  print('negative f1: %f' % negEff1)
