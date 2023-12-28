import torch
import torch.nn as nn

## 2 Layer MLP
# Try to use equalLinear Layer from utils

class CanonicalEncoder(nn.Module):
    def __init__(self, opts):
        super(CanonicalEncoder, self).__init__()
        input_size = opts.canonical_encoder_input_size
        hidden_size = opts.canonical_encoder_hidden_size
        output_size = opts.canonical_encoder_output_size
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2

    def forward(self, x):
        with torch.autograd.profiler.record_function('input'):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

        return x

if __name__ == "__main__":
    from Options.BaseOptions import opts
    ce = CanonicalEncoder(opts).to(opts.device)
    z_s = torch.randn([100, 18 * 512]).to(opts.device)

    # for batch in z_s:
    #     print(batch.shape)
    z_s_c = ce(z_s)
    print(z_s_c.shape)
    # for p in model.parameters():
    #     print(p.shape)