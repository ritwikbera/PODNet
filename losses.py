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

def TSLoss():     
    pass
    #L_TS += - 0.0*((c_t[:,1:,:]*c_t[:,:-1,:])*mask[:,1:,:]).sum(-1).sum(-2).mean()