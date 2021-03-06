import torch
from torch import Tensor, nn
import torch.nn.functional as F 
from functools import partial

EPS = 1e-17
NEG_INF = -1e30

def class_weights(targets):
    num_classes = targets.size(-1)
    targets = targets.view(-1, num_classes)
    class_weights = Tensor([1]).repeat(num_classes)
    # print(targets)
    for i in range(len(class_weights)):
        try:
            class_weights[i] = 10.0/len(targets[targets[:,i] == 1])
        except ZeroDivisionError: #to tackle non-occuring classes
            class_weights[i] = 10.0
        # print(class_weights[i])

    return class_weights

def weighted_BCE(outputs, targets, mask=None, weights=None):
    
    #Note: BCEWithLogits avoided the numerical instability (nan loss) problem seen with softmax
    if weights is not None:
        # assert len(weights) == 2 #different weighting for action present or absent
        
        loss = weights*F.binary_cross_entropy_with_logits(outputs,targets,reduction='none')
    else:
        loss = F.binary_cross_entropy_with_logits(outputs,targets,reduction='none')

    # print(outputs)

    if mask is not None:
        BCE_loss = (loss*mask).sum(-1).sum(-2).mean()

    # print(BCE_loss)

    return BCE_loss

def focal_loss(outputs, targets, mask=None, alpha=1.0, gamma=4):
    #focal loss parameters, borrowed from RetinaNet
    alpha = 1.0 
    gamma = 4

    BCE_loss = F.binary_cross_entropy_with_logits(outputs,targets,reduction='none')
    pt = torch.exp(-BCE_loss)
    # print(pt) #print confidence scores
        
    F_loss = (alpha*(1-pt)**gamma)*BCE_loss

    if mask is None:
        mask = torch.ones(*F_loss.size())
    
    #average across batch size, sum across sequence length
    masked_F_loss = (F_loss*mask).sum(-1).sum(-2).mean() 

    return masked_F_loss


    
def DynamicsLoss(next_state_segment, next_state_pred, PAD_TOKEN, device):
    mask = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor).to(device)
    L_ODC = (((next_state_segment - next_state_pred)**2)*mask).sum(-1).sum(-2).mean()
    return L_ODC

def BCLoss(action_segment, action_pred, PAD_TOKEN, device, use_discrete=False, use_focal=False, use_weighting=True, weights=None):

    mask = (action_segment!=PAD_TOKEN).type(torch.LongTensor).to(device)
    
    if use_discrete:

        #clean out PAD_TOKENS before entering action_segment into BinaryCrossEntropy
        pad_mask = (1-mask).type(torch.BoolTensor)
        action_segment.masked_fill_(pad_mask, 0)
        assert ((action_segment>=0).all()  and (action_segment<=1).all()) #unit test to ensure all inputs to BCE are >0, <1

        if use_focal:
            L_BC = focal_loss(action_pred, action_segment, mask)
        else:
            if use_weighting:
                # weights = class_weights(action_segment)
                L_BC = weighted_BCE(action_pred, action_segment, mask, weights)
            else:
                L_BC = weighted_BCE(action_pred, action_segment, mask)
    else:   
        L_BC = (F.mse_loss(action_pred, action_segment, reduction='none')*mask).sum(-1).sum(-2).mean()
    
    return L_BC

def kl_gaussian(mu, log_var):
    """KL divergence between Gaussian posterior and standard normal prior."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = -1)

def kl_categorical_uniform(preds):
    """KL divergence between categorical distribution and uniform prior."""
    kl_div = preds * torch.log(preds + EPS)  # Constant term omitted.
    return kl_div.sum(-1)

def KLDLoss(logits, mask, args, device):
    mask = mask.to(device)

    # Evaluate KL Divergence loss, and sum over masked length, avg over batch.
    
    if args.latent == 'gaussian':
        mu, log_var = torch.split(logits, args.latent_dim, dim=-1)
        mu = torch.clamp(mu, min=-1, max=1)
        log_var = torch.clamp(log_var, min=0.1, max=10)
        kl_z = kl_gaussian(mu, log_var)
    elif args.latent == 'concrete':
        kl_z = kl_categorical_uniform(F.softmax(logits, dim=-1))
    else:
        raise ValueError('Invalid argument for `latent_dist`.')
    
    kl_z = (kl_z*mask).sum(1).mean()
    
    return kl_z

def TSLoss(c_t, mask, args, device):  
    mask = mask.to(device)
    ind = torch.argmax(c_t, dim=-1, keepdim=True)
    o_t = torch.FloatTensor(c_t.shape).zero_()
    o_t = o_t.scatter_(-1, ind, 1)
    L_TS = (1-((o_t[:,1:,:]*o_t[:,:-1,:]).sum(-1))*mask[:,1:]).sum(-2).mean()
    return L_TS

if __name__ == '__main__':
    print(TSLoss(torch.rand(2,40,3), torch.ones(2,40), device='cpu'))