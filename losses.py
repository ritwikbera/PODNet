import torch
from torch import Tensor, nn
import torch.nn.functional as F 

def DynamicsLoss(next_state_segment, next_state_pred, PAD_TOKEN, device):
    mask = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor).to(device)
    L_ODC = (((next_state_segment - next_state_pred)**2)*mask).sum(-1).sum(-2).mean()
    return L_ODC

def BCLoss(action_segment, action_pred, PAD_TOKEN, device, use_discrete=False):
    #focal loss parameters, borrowed from RetinaNet
    alpha = 1.0 
    gamma = 4

    mask = (action_segment!=PAD_TOKEN).type(torch.LongTensor).to(device)
    
    #clean out PAD_TOKENS before entering action_segment into BinaryCrossEntropy
    pad_mask = (1-mask).type(torch.BoolTensor)
    action_segment.masked_fill_(pad_mask, 0)

    # print(action_segment)
    assert ((action_segment>=0).all()  and (action_segment<=1).all()) #unit test to ensure all inputs to BCE are >0, <1

    if use_discrete:
        BCE_loss = F.binary_cross_entropy(action_pred,action_segment,reduction='none')
        pt = torch.exp(-BCE_loss)
        # print(pt) #print confidence scores
        
        F_loss = (alpha*(1-pt)**gamma)*BCE_loss
        # F_loss = BCE_loss # Uncomment this line to disable focal loss
        
        L_BC = (F_loss*mask).sum(-1).sum(-2).mean()
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
    ind = torch.argmax(c_t, dim=-1, keepdim=True)
    o_t = torch.FloatTensor(c_t.shape).zero_()
    o_t = o_t.scatter_(-1, ind, 1)
    L_TS = (1-((o_t[:,1:,:]*o_t[:,:-1,:]).sum(-1))*mask[:,1:]).sum(-2).mean()
    return L_TS

if __name__ == '__main__':
    print(TSLoss(torch.rand(2,40,3), torch.ones(2,40), device='cpu'))