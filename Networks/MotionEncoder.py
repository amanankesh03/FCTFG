import torch
import torch.nn as nn
import math
## 3 Layer MLP
# Try to use equalLinear Layer from utils

class MotionEncoder(nn.Module):
    def __init__(self, opts):
        super(MotionEncoder, self).__init__()

        # n_styles = int(math.log(opts.size, 2) * 2 - 2)
        input_neurons = 2 * opts.latent_dim
        hidden_neurons = opts.latent_dim
        output_neurons = opts.latent_dim
        
        self.fc1 = nn.Linear(input_neurons, hidden_neurons)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)  # Fully connected layer 2
        self.fc3 = nn.Linear(hidden_neurons, output_neurons)  # Fully connected layer 3

        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


if __name__ == "__main__":
    from Options.BaseOptions import opts
    me = MotionEncoder(opts)
    n_styles = int(math.log(opts.size, 2) * 2 - 2)
    
    z_a_d = torch.randn([100, (n_styles + 1) * 512])
    z_c_d = me(z_a_d)
    print(z_c_d.shape)
    z_c_d = z_c_d.view(z_c_d.shape[0], -1, 512)
    print(z_c_d.shape)
    for p in me.parameters():
        print(p.shape)