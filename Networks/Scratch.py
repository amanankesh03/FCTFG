file = '/home/zottang/Data/training_data/training_data/trainset-eikitai1/How to TRICK Your Interviewer into Hiring You_clip0022_000_with_audio.mp4'


import subprocess
import torch
from torchvision.io.video import read_video, write_video

def convert_video_fps(input_file, output_file, target_fps):
    # Use FFmpeg to convert video FPS
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_file,
        '-filter:v', f'fps={target_fps}',
        '-c:a', 'copy',
        '-strict', 'experimental',
        output_file
    ]

    subprocess.run(ffmpeg_command, check=True)

def read_video_tensor(file_path):
    # Read video file into a PyTorch tensor
    video, audio, info = read_video(file_path)
    return video, audio, info

# Specify the file paths

output_file = './tmp.mp4'

# Convert video FPS using FFmpeg
target_fps = 25
convert_video_fps(file, output_file, target_fps)

# Read the converted video into a PyTorch tensor
video_tensor, _, _ = read_video_tensor(output_file)

# Print the shape of the video tensor
print("Video Tensor Shape:", video_tensor.shape)
