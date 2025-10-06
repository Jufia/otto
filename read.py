import json
import os
import pandas as pd
import numpy as np
import polars as pl
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, type='train'):
        super().__init__()
        self.type = type
        self.ID2indx = pd.Series()
        self.indx2ID = pd.Series()

        ids, ts, labels, max_len = self.read_json(type)
        self.ids = ids
        self.ts = ts
        self.labels = labels
        self.max_len = max_len

        self.len = len(ids)
        self.item_number = len(self.ID2indx)

    def __getitem__(self, index):
        self.pad_id = 0
        length = len(self.ids[index])

        ids = self.ids[index]
        seq = self.ID2indx[ids].values.tolist() + [self.pad_id] * (self.max_len-length)
        time_stemp = self.ts[index] + [0] * (self.max_len-length)
        label = self.labels[index] + [0] * (self.max_len-length)
        padding_mask = [0]*length + [1]*(self.max_len-length)

        return seq, label, padding_mask


    def __len__(self):
        return self.len
    
    def get_info(self):
        print(f'''总共有{self.len}个交互序列，
              最长的序列为{self.max_len}, 
              总共有{self.item_number}篇article''')

    def read_json(self, type):
        path = f'./dataset/{type}.jsonl'

        type_labels = {'clicks':1, 'carts':2, 'orders':3}
        sample_size = 1_000

        chunks = pd.read_json(path, lines=True, chunksize = sample_size)
        
        ids = []
        labels = []
        time_stamp = []
        max_len = 0
        for c in chunks:
            data = c['events']
            for line in data:
                if type == 'test' and len(line) < 2:
                    continue

                line = pd.DataFrame(line)

                id = line['aid'].to_list()
                ids.append(id)

                lb = line['type'].to_list()
                label = [type_labels[type] for type in lb]
                labels.append(label)

                ts = line['ts'].to_list()
                time = [n-ts[0] for n in ts]
                time_stamp.append(time)

                max_len = max(max_len, len(line))

            break
        
        unique_id = np.unique(np.concatenate(ids))
        idx = range(1, len(unique_id)+1)
        self.indx2ID = pd.Series(unique_id, index=idx)
        self.ID2indx = pd.Series(idx, index=unique_id)

        return ids, time_stamp, labels, max_len
