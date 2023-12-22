import torch
import torch.nn as nn

## 2 Layer MLP
# Try to use equalLinear Layer from utils

class CanonicalEncoder(nn.Module):
    def __init__(self, opts):
        input_size = opts.canonical_encoder_input_size
        hidden_size = opts.canonical_encoder_hidden_size
        output_size = opts.canonical_encoder_output_size
        super(CanonicalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    input_size = 10
    hidden_size = 10
    output_size = 10
    model = CanonicalEncoder(input_size, hidden_size, output_size)

    for p in model.parameters():
        print(p.shape)