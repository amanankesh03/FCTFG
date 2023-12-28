import torch
import torch.nn as nn

## 3 Layer MLP
# Try to use equalLinear Layer from utils

class MotionEncoder(nn.Module):
    def __init__(self, opts):
        super(MotionEncoder, self).__init__()

        input_size = opts.motion_encoder_input_size
        hidden_sizes = opts.motion_encoder_hidden_sizes #{}
        output_size = opts.motion_encoder_output_size
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # Fully connected layer 2
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)  # Fully connected layer 3

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
    z_a_d = torch.randn([100, 2 * 18 * 512])
    z_c_d = me(z_a_d)
    print(z_c_d.shape)
    z_c_d = z_c_d.view(z_c_d.shape[0], -1, 512)
    print(z_c_d.shape)
    for p in me.parameters():
        print(p.shape)