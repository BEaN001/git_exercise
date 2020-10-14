import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import time
import random

torch.backends.cudnn.deterministic = True

# General Settings

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 20000
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 1

# Dataset
TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)
# LABEL = data.LabelField()
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(RANDOM_SEED),
                                          split_ratio=0.8)

print(f'Num Train: {len(train_data)}')
print(f'Num Valid: {len(valid_data)}')
print(f'Num Test: {len(test_data)}')












































