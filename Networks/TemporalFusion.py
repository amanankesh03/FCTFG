import torch
import torch.nn as nn

## single 1D convolution 
 
class TemporalFusion(nn.Module):
    def __init__(self, opts):
        super(TemporalFusion, self).__init__()
        input_size = opts.temporal_fusion_input_size
        output_size = opts.temporal_fusion_output_size
        kernel_size = opts.temporal_fusion_kernel_size
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        x = self.conv1d(x)
        return x
    
if __name__ == "__main__":
    # Instantiate the model
    from Options.BaseOptions import opts
    input_size = 36 * 512 # Number of input channels/feature
    output_size = 18 * 512 # Number of output channels/features
    kernel_size = 3
    model = TemporalFusion(opts)

    # Dummy input data
    dummy_input = torch.randn((1, 36, 512))  # Batch size of 32

    # Forward pass
    output = model(dummy_input)

    # Print model summary
    print(model)
    print("Input size:", input_size)
    print("Output size:", output_size)
    print("Output shape:", output.shape)
