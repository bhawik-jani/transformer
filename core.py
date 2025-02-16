import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size,T,_ = x.shape

        q = self.w_q(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k)**0.5
        if mask is not None:
            scores = scores.masked_fill(mask[:T,:T] == 0, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
        return self.fc_out(output)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, mask=None):
        tgt2 = self.self_attn(self.norm1(tgt), mask=mask)
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.ffn(self.norm2(tgt))
        tgt = tgt + self.dropout(tgt2)
        return tgt
    
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # Final normalization

    def forward(self, tgt, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, mask=mask)
        return self.norm(tgt)