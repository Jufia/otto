import torch
import torch.nn as nn

class embedding_model(nn.Module):
    def __init__(self, item_number, item_dim, action_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(item_number, item_dim)
        self.actn_embedding = nn.Embedding(4, action_dim)

        


