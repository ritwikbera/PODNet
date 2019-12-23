import torch
from torch.nn import functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.size()).to(device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, latent_dim, categorical_dim, temperature=0.1, hard=False, device='cpu'):
    y = gumbel_softmax_sample(logits, temperature, device)
    
    if not hard:
        return y.view(*y.size()[:-2], latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(*y.size()[:-2], latent_dim * categorical_dim)