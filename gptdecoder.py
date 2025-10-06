import torch
import torch.nn as nn

class GPTDecoder(nn.Module):
    def __init__(self, num_item, item_dim, action_dim, max_len=200):
        super().__init__()
        d_model = item_dim + action_dim
        self.item_embedding = nn.Embedding(num_item, item_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(4, action_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2,
            norm=nn.LayerNorm(d_model),
        )

        self.fc = nn.Linear(d_model, num_item)

    def forward(self, seq, action, padding_mask=None, casual_mask=None):
        b, l = seq.shape

        positions = torch.arange(l, device=seq.device)
        seq_emb = self.item_embedding(seq)
        act_emb = self.action_embedding(action)

        if padding_mask == None:
            padding_mask = (seq_emb == 0)

        if casual_mask == None:
            casual_mask = torch.triu(torch.ones(l, l), diagonal=1).bool()

        input = torch.concat([seq_emb, act_emb], dim=-1) + self.pos_emb(positions)
        output = self.block(input, src_key_padding_mask=padding_mask.bool(), mask=casual_mask.bool())

        output = self.fc(output)

        return output

if __name__ == '__main__':
    m = GPTDecoder(10, 32, 32)
    x = torch.LongTensor([[2, 2, 4], [4, 6, 0]])
    action = torch.LongTensor([[1, 2, 1], [0, 1, 1]])
    mask = torch.LongTensor([[False, True, True], [False, False, False]])
    out = m(x, action, mask)
    print(out.shape)
