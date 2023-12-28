import torch
from Options.BaseOptions import opts
import torch.autograd.profiler as profiler
from Networks.CanonicalEncoder import CanonicalEncoder

# Enable the profiler

ce = CanonicalEncoder(opts).to(opts.device)
with profiler.profile(record_shapes=True, use_cuda=False) as prof:
    # Your code here
    input_data = torch.randn(1, 18 * 512).to(opts.device)
    output_data = ce(input_data)

# Print the profiling results
print(prof.key_averages().table())
