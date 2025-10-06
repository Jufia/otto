import numpy as np
import torch
import pickle
from tqdm import tqdm
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from collections import defaultdict
from collections import Counter
import gc
import glob

import read
from torch.utils.data import DataLoader
from gptdecoder import GPTDecoder

data = read.MyDataset('test')
data.get_info()
dataloader = DataLoader(data, batch_size=12, shuffle=True)

m = GPTDecoder(num_item=data.item_number, item_dim=32, action_dim=32, max_len=data.max_len)
casual_mask = torch.triu(torch.ones(data.max_len, data.max_len), diagonal=1).bool()
for d in dataloader:
    seq, action, padding_mask = d
    seq, action, padding_mask = np.array(seq).transpose(1, 0), np.array(action).transpose(1, 0), np.array(padding_mask).transpose(1, 0)

    seq, action, padding_mask = torch.LongTensor(seq), torch.LongTensor(action), torch.LongTensor(padding_mask).bool()

    logits = m(seq, action, padding_mask)
