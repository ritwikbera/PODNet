import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F

dtype = torch.float
device = torch.device("cpu")

state_dim = 3
option_dim = 3
action_dim = 3
mlp_in = state_dim + option_dim
mlp_hidden = mlp_in//2+1
mlp_out = action_dim
N, D_in, H, D_out = 100, mlp_in, mlp_hidden, mlp_out

learning_rate = 1e-4
iterations = 10000

model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
        torch.nn.ReLU(), torch.nn.Linear(H, D_out),)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


x = np.array([i*np.ones(D_in) for i in range(N)])
y = np.array([i*np.ones(D_out) for i in range(N)])

x = torch.tensor(x, device=device, dtype=dtype)
y = torch.tensor(y, device=device, dtype=dtype)
print(x.size(), y.size())

for t in range(iterations):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save({'model_state_dict':model.state_dict()}, 'mlp_checkpoint.pth')
checkpoint = torch.load('mlp_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(model(x[10]))