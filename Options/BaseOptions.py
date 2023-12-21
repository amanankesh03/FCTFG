import argparse
parser = argparse.ArgumentParser()

### Training Options ####
parser.add_argument("--iter", type=int, default=800000, help='Number of image iterations')
parser.add_argument("--size", type=int, default=256, help='Image size')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--start_iter", type=int, default=0)
parser.add_argument("--resume_ckpt", type=str, default=None)
parser.add_argument("--port", type=str, default='12345')
parser.add_argument("--addr", type=str, default='localhost')
parser.add_argument("--exp_path", type=str, default='./exps/')
parser.add_argument("--exp_name", type=str, default='v1')
parser.add_argument("--save_freq", type=int, default=1000)
parser.add_argument("--display_freq", type=int, default=5000)
parser.add_argument("--dataset", type=str, default='./data')

### Visual Encoder Options ###


### Audio Encoder Options ###


### Canonical Encoder Options ###


### Motion Encoder Options ###


### Temporal Fusion Options ###


### Generator Options ###

parser.add_argument("--g_reg_every", type=int, default=4, help='generator regularization every nth iteration')
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--channel_multiplier", type=int, default=1)
parser.add_argument("--latent_dim_style", type=int, default=512)
parser.add_argument("--latent_dim_motion", type=int, default=20)

### Discriminator Options ###

parser.add_argument("--d_reg_every", type=int, default=16, help='discriminator regularization every kth iteration')


opts = parser.parse_args()
