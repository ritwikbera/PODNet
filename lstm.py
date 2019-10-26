import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

input_size = 20
# If `per_element` is True, then LSTM reads in one timestep at a time.
per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
# size of hidden layers
h1 = 32
output_dim = 1
num_layers = 2
learning_rate = 1e-3
num_epochs = 500
dtype = torch.float

input_dim = 1
N = 200

X_train = np.array([i*np.ones(input_dim) for i in range(N)])
X_test = X_train
y_train = np.array([i*np.ones(output_dim) for i in range(N)])
y_test = y_train

X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor).view(-1)
y_test = torch.from_numpy(y_test).type(torch.Tensor).view(-1)

X_train = X_train.view([input_size, -1, 1])
X_test = X_test.view([input_size, -1, 1])

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
        num_layers=2):
        super(LSTM, self).__init()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


model = LSTM(lstm_input_size, h1, batch_size=num_train,
    output_dim=output_dim, num_layers=num_layers)

loss_fn = nn.MSELoss(size_average = False)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
hist = np.zeros(num_epochs)

for t in range(num_epochs):
    model.zero_grad()
    model.hidden = model.init_hidden()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()