import torch
from torch import nn, Tensor            
import torch.nn.functional as F

class probs(nn.Module):
    def __init__(self, latent_dim, categorical_dim):
        super(probs, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, q, tau):
        if not self.categorical_dim == 1: # means not continuous variable
            q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
            q_y = F.gumbel_softmax(q_y, tau=tau, hard=not self.training)
            c_t = q_y.view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)
        else :
            mu, logvar = torch.split(q, self.latent_dim, dim=-1)
            mu = torch.clamp(mu, min=-1, max=1)
            logvar = torch.clamp(logvar, min=0.1, max=10)
            c_t = self.reparameterize(mu, logvar)

        return c_t