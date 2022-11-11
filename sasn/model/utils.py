import torch

from config import settings
from components.vocab import SOS_token, EOS_token

def tensor_from_sequence(vocab, sequence):
    indeces = [vocab.word2idx[word] for word in sequence]
    indeces.append(EOS_token)
    return torch.tensor(indeces, dtype=torch.long, device=settings.DEVICE).view(-1, 1)
