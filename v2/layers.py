import torch
from torch import Tensor, nn 
import torch.nn.functional as F 
from torch.autograd import Variable
import math
import numpy as np 

SEGMENT_SIZE = 128
BATCH_SIZE = 1
INPUT_SIZE = 2 #2d state test

class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, max_seq_len = SEGMENT_SIZE):
        super().__init__()
        self.input_dim = input_dim
        pe = torch.zeros(max_seq_len, input_dim)
        for pos in range(max_seq_len):
            for i in range(0, input_dim, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/input_dim)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/input_dim)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.input_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len],requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, att_dim=5, heads=1):
        super().__init__()
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.output_dim = output_dim
        self.d_k = att_dim // heads
        self.h = heads
        
        self.q_linear = nn.Linear(input_dim, att_dim)
        self.v_linear = nn.Linear(input_dim, att_dim)
        self.k_linear = nn.Linear(input_dim, att_dim)
        self.out = nn.Linear(att_dim, output_dim)

    def attention(self, q, k, v, d_k, mask=None):
        logits = torch.matmul(q, k.transpose(-2, -1))

        # print('Logits size {}'.format(logits.size()))
        # print('Mask size {}'.format(mask.size()))
        # print('Value matrix size {}'.format(v.size()))

        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
            probs = F.softmax(logits, dim=-1)

        # print('Probs size {}'.format(probs.size()))

        output = torch.matmul(probs, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        context = self.attention(q, k, v, self.d_k, mask) 
        # print('Context vector size {}'.format(context.size()))
        concat = context.transpose(1,2).contiguous().view(bs, -1, self.att_dim)
        # print('Concatenated context vector size {}'.format(concat.size()))
        output = self.out(concat)
        return output

if __name__=='__main__':
    a = torch.arange(SEGMENT_SIZE)
    mask = (a[None, :] <= a[:, None]).type(torch.FloatTensor)
    mhatt = MultiHeadAttention(2,3)
    x = torch.randn(BATCH_SIZE, SEGMENT_SIZE, INPUT_SIZE)
    out = mhatt(x,x,x, mask)
    print('MHAtt Output Size {}'.format(out.size()))