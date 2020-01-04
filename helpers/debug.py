import torch
from torch import Tensor, nn
import torch.nn.functional as F 
import pdb

class Hook():
    def __init__(self, module, backward=False):
        self.backward = backward
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

        if not self.backward:
            print('Input Tensor to {} is: {} \n'.format(module.__class__.__name__, self.input))
        else:
            print('Backpropagated gradient to {} is {} \n'.format(module.__class__.__name__, self.output))
        
        pdb.set_trace()
    
    def close(self):
        self.hook.remove()