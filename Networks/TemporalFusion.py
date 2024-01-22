import torch
import torch.nn as nn
import math
## single 1D convolution 
 
class TemporalFusion(nn.Module):
    def __init__(self, opts):
        super(TemporalFusion, self).__init__()
        
        n_styles = int(math.log(opts.size, 2) * 2 - 2)

        input_channels = n_styles 
        output_channels = n_styles
        kernel_size = opts.temporal_fusion_kernel_size
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        x = self.conv1d(x)
        return x
    
if __name__ == "__main__":
    # Instantiate the model
    from Options.BaseOptions import opts
    model = TemporalFusion(opts)

    # Dummy input data
    n_styles = int(math.log(opts.size, 2) * 2 - 2)
    in_ch = n_styles * opts.num_frames
    dummy_input = torch.randn((1, in_ch, opts.latent_dim))  # Batch size of 32

    # Forward pass
    output = model(dummy_input)

    # Print model summary
    print(model)

    print("Output shape:", output.shape)
