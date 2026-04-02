import torch
from torch import nn

class BiRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiRNN with bidirectional=True
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Vì là bidirectional nên hidden_size của đầu ra sẽ nhân 2
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # init hidden state (cần nhân 2 số lượng layer cho 2 chiều)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, h0)
        
        out = out[:, -1, :]
        out = self.fc(out)
        return out