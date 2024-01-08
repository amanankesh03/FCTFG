import torch
import torch.nn as nn
import math
## 2 Layer MLP
# Try to use equalLinear Layer from utils

class CanonicalEncoder(nn.Module):
    def __init__(self, opts):
        super(CanonicalEncoder, self).__init__()
        n_styles = int(math.log(opts.size, 2) * 2 - 2)

        num_neurons = n_styles * opts.latent_dim
        self.fc1 = nn.Linear(num_neurons, num_neurons)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(num_neurons, num_neurons)  # Fully connected layer 2

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    from Options.BaseOptions import opts
    ce = CanonicalEncoder(opts).to(opts.device)
    n_styles = int(math.log(opts.size, 2) * 2 - 2)
    
    z_s = torch.randn([100, n_styles * 512]).to(opts.device)

    # for batch in z_s:
    #     print(batch.shape)
    z_s_c = ce(z_s)
    print(z_s_c.shape)
    # for p in model.parameters():
    #     print(p.shape)