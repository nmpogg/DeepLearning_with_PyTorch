import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # for classification
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # hidden state t = 0 
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)

        # cell state t = 0
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)

        out, hT = self.lstm(X, (h0, c0))

        out = out[:, -1, :] # output at t = T
        out = self.fc(out)

        return out
