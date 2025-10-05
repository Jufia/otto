import json
import os
import pandas as pd
import polars as pl
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, type='train'):
        super().__init__()
        self.type = type

        ids, ts, labels, max_len = self.read_json(type)
        self.ids = ids
        self.ts = ts
        self.labels = labels
        self.max_len = max_len
        self.len = len(ids)

    def __getitem__(self, index):
        self.pad_id = self.max_id + 1
        length = len(self.ids[index])

        seq = self.ids[index] + [self.pad_id] * (self.max_len-length)
        time_stemp = self.ts[index] + [0] * (self.max_len-length)
        label = self.labels[index] + [3] * (self.max_len-length)

        return seq, label


    def __len__(self):
        return self.len
    
    def get_info(self):
        print(f'总共有{self.len}个序列，最长的序列为{self.max_len}, 原始id中最大的为{self.max_id}')

    def read_json(self, type):
        path = f'./dataset/{type}.jsonl'
        self.max_id = -float('inf')

        type_labels = {'clicks':0, 'carts':1, 'orders':2}
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
                self.max_id = max(self.max_id, max(id))
                ids.append(id)

                lb = line['type'].to_list()
                label = [type_labels[type] for type in lb]
                labels.append(label)

                ts = line['ts'].to_list()
                time = [n-ts[0] for n in ts]
                time_stamp.append(time)

                max_len = max(max_len, len(line))

            break

        return ids, time_stamp, labels, max_len
