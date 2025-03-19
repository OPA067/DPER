import torch
import torch.nn as nn
import torch.nn.functional as F

from models.local_feat_agg import MultiHeadSelfAttention


class PER_Net(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.embed_dim = d_in

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.fc1 = nn.Linear(d_in, d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, local_feat, global_feat, out, pad_mask=None):

        residual, attn = self.attention(local_feat, pad_mask)

        fc_out = self.fc1(global_feat)
        global_feat = self.fc(residual) + fc_out

        mu = l2_normalize(out)
        sigma = global_feat
        return {
            'mu': mu,
            'sigma': sigma,
        }

def l2_normalize(tensor, axis=-1):

    return F.normalize(tensor, p=2, dim=axis)