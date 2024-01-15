import torch
import subprocess
from torchvision.io.video import read_video
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

def convert_video_to_tensor(input_file, target_fps, h, w):
    # Use FFmpeg to read video frames and set the output frame rate
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_file,
        '-r', str(target_fps),
        '-f', 'image2pipe',  # Output raw image data to pipe
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-loglevel', 'quiet',
        '-'
    ]

    # Execute FFmpeg command and capture its output as a byte stream
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE)
    video_frames, _ = ffmpeg_process.communicate()

    # Convert the byte stream to a PyTorch tensor
    video_tensor = torch.from_numpy(
        torch.frombuffer(video_frames, dtype=torch.uint8).numpy().reshape(-1, h, w, 3)
    )

    return video_tensor


def change_video_fps(input_file, output_file, video_fr):
# Use FFmpeg to convert video FPS directly during conversion
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_file,
        '-r', str(video_fr),  # Set the output frame rate
        '-c:a', 'copy',
        '-strict', 'experimental',
        output_file
    ]

    subprocess.run(ffmpeg_command, check=True)


def change_frame_rate(self, video, info, target_frame_rate):
    
    original_frame_rate = info['video_fps']
    frame_rate_ratio = original_frame_rate / target_frame_rate
    new_num_frames = int(video.size(0) / frame_rate_ratio)
    print('change frame rate : ', video.shape, info)
    resized_video = F.resize(video, (new_num_frames,), InterpolationMode.NEAREST)

    return resized_video


# Specify the file path
# input_file = '/path/to/your/input_video.mp4'

# # Convert video to PyTorch tensor
# target_fps = 25
# video_tensor = convert_video_to_tensor(input_file, target_fps)

# # Print the shape of the video tensor
# print("Video Tensor Shape:", video_tensor.shape)
