import torch

# Create a tensor 'ws'
ws = torch.randn(5, 10)  # Assuming a 2D tensor with shape (5, 10)

# Define a block with information about the block to be extracted
class BlockInfo:
    def __init__(self, num_conv, num_torgb):
        self.num_conv = num_conv
        self.num_torgb = num_torgb

# Assume you have a 'BlockInfo' instance
block = BlockInfo(num_conv=3, num_torgb=2)

# Index to start extracting the block
w_idx = 2

# Extract the block using 'narrow'
block_ws = []
block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))

# Print the original tensor and the extracted block
print("Original ws:")
print(ws.shape)
print("\nExtracted Block:")
print(block_ws[0].shape)
