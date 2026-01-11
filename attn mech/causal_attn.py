import torch
import torch.nn as nn



inputs = torch.tensor(
  [[0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)

d_in = inputs.shape[1] 
d_out = 2 


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

   

sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs) 
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)
attn_scores = queries @ keys.T


context_length = attn_scores.shape[0]

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)            
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)                       # Create upper triangular mask with -inf values
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)           # doing normalization and get 0 at upper triangular part
print(attn_weights)

dropout = torch.nn.Dropout(0.5)                                                        #dropout to make nural active at 50%
attn_weight_dropout = dropout(attn_weights)
print(attn_weight_dropout)


context_vec = attn_weight_dropout @ values
print(context_vec)

