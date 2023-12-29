import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define a simple 1D convolutional layer
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)

# Create a 1D input signal
input_signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).view(1, 1, -1)  # Shape: [batch_size, channels, sequence_length]

# Perform 1D convolution
output_signal = conv1d(input_signal)

# Convert tensors to numpy arrays for plotting
input_signal_np = input_signal.squeeze().numpy()
output_signal_np = output_signal.squeeze().detach().numpy()

# Plot the input and output signals
plt.figure(figsize=(10, 4))

# Plot input signal
plt.subplot(2, 1, 1)
plt.plot(input_signal_np, marker='o')
plt.title('Input Signal')
plt.xlabel('Index')
plt.ylabel('Amplitude')

# Plot output signal
plt.subplot(2, 1, 2)
plt.plot(output_signal_np, marker='o')
plt.title('Output Signal after 1D Convolution')
plt.xlabel('Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
