import torch
from torch import Tensor, nn 
import torch.nn.functional as F 
from torch.autograd import Variable
import math
import numpy as np 

class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_hidden=32, dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, mlp_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.linear_3 = nn.Linear(mlp_hidden, output_dim)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.dropout(F.relu(self.linear_2(x)))
        x = self.linear_3(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
 
    def forward(self, x):
        seg_length = x.size(-2)
        pe = torch.zeros(seg_length, 1).to(self.device)

        for pos in range(x.size(-2)):
            pe[pos,:] = math.sin((pos/(seg_length*10))*math.pi/2)

        return torch.cat((x, pe.repeat(*x.size()[:-2],1,1)), dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_embed_dim, output_dim=12, heads = 2):
        super().__init__()
        
        self.input_embed_dim = input_embed_dim
        self.output_dim = output_dim
        self.d_k = output_dim // heads
        self.h = heads
        
        self.q_linear = nn.Linear(input_embed_dim, output_dim)
        self.v_linear = nn.Linear(input_embed_dim, output_dim)
        self.k_linear = nn.Linear(input_embed_dim, output_dim)
        self.out = nn.Linear(output_dim, input_embed_dim)

    def attention(self, q, k, v, d_k, mask=None):
        logits = torch.matmul(q, k.transpose(-2, -1))

        # print('Logits size {}'.format(logits.size()))
        # print('Mask size {}'.format(mask.size()))
        # print('Value matrix size {}'.format(v.size()))

        if mask is not None:
            mask = mask.repeat(*mask.size()[:-2],1,1)
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
        concat = context.transpose(1,2).contiguous().view(bs, -1, self.output_dim)
        # print('Concatenated context vector size {}'.format(concat.size()))
        output = self.out(concat)
        return output

if __name__=='__main__':
    # BATCH_SIZE = 1
    # SEGMENT_SIZE = 512
    # a = torch.arange(SEGMENT_SIZE)
    # mask = (a[None, :] <= a[:, None]).type(torch.FloatTensor)
    # mhatt = MultiHeadAttention(2,3)
    # x = torch.randn(BATCH_SIZE, SEGMENT_SIZE, INPUT_SIZE)
    # out = mhatt(x,x,x, mask)
    # print('MHAtt Output Size {}'.format(out.size()))

    pos_enc = PositionalEncoder(512)
    x = torch.zeros(3,512,2)
    print(pos_enc(x).size())
    print(pos_enc(x)[:,:10])
    print('Last 10')
    print(pos_enc(x)[:,-10:])