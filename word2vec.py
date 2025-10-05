import polars as pl
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from collections import defaultdict
from collections import Counter
import gc
import glob

import read

data = read.MyDataset('test')
data.get_info()
dataloader = read.DataLoader(data, batch_size=12, shuffle=True)

w2vec = Word2Vec(sentences=data.ids, vector_size=50, epochs=10, sg=1, window=5, sample=1e-3, ns_exponent=1, min_count=1, workers=30)

aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}