import torch
from torch import Tensor, nn
import torch.nn.functional as F 

def DynamicsLoss(next_state_segment, next_state_pred, PAD_TOKEN, device):
    mask = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor).to(device)
    L_ODC = (((next_state_segment - next_state_pred)**2)*mask).sum(-1).sum(-2).mean()
    return L_ODC

def BCLoss(action_segment, action_pred, PAD_TOKEN, device, use_discrete=False):
    mask = (action_segment!=PAD_TOKEN).type(torch.FloatTensor).to(device)
    if use_discrete:
        L_BC = (F.binary_cross_entropy(action_pred,action_segment,reduction='none')*mask).sum(-1).sum(-2).mean()
    else:   
        L_BC = (((action_segment - action_pred)**2)*mask).sum(-1).sum(-2).mean()
    return L_BC

def KLDLoss(qy, mask, categorical_dim, device):
    mask = mask.to(device)
    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = (torch.sum(qy * log_ratio, dim=-1)*mask).sum(-2).mean()
    return KLD 

def TSLoss(c_t, mask, device):  
    mask = mask.to(device)   
    L_TS = -((c_t[:,1:,:]*c_t[:,:-1,:]).sum(-1)*mask[:,1:]).sum(-2).mean()
    return L_TS