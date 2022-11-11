from collections import Counter

from config import settings

SOS_token = 0
EOS_token = 1

# --------------------
# Generator component.
# --------------------

class VocabGenerator:
    def __init__(self):
        self.counter = Counter()

    def add(self, tokens):
        self.counter.update(tokens)

    def generate(self):
        tokens = list()
        # Limit vocabulary size.
        for token, count in self.counter.most_common(settings.MAX_VOCAB_SIZE):
            if count < settings.FREQUENCY_CUTOFF:
                break
            tokens.append(token)
        return Vocab(tokens)

# ----------------------------------------------------
# A vocabulary is a mapping between words and indices.
# ----------------------------------------------------

class Vocab:
    def __init__(self, tokens):
        self.word2idx = dict()
        self.idx2word = dict()
        for idx, word in enumerate(['<SOS>', '<EOS>'] + tokens):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)
