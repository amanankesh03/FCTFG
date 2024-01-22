import torch
import subprocess
# from torchvision.io.video import read_video
from torchvision.io import read_video, write_video
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


def change_frame_rate(video, info, target_frame_rate):
    
    original_frame_rate = info['video_fps']
    frame_rate_ratio = original_frame_rate / target_frame_rate
    new_num_frames = int(video.size(0) / frame_rate_ratio)
    print('change frame rate : ', video.shape, info)
    resized_video = F.resize(video, (new_num_frames,), InterpolationMode.NEAREST)

    return resized_video

def interpolate_frames(frame1, frame2, t):
    # Linear interpolation between two frames
    return (1 - t) * frame1 + t * frame2



def interpolate_frames(frame1, frame2, t):
    return (1 - t) * frame1 + t * frame2

def adjust_fps(frames, info, target_fps):

    original_fps = info['video_fps']

    adjustment_factor =  target_fps / original_fps

    if adjustment_factor > 1:
     
        indices_to_interpolate = torch.arange(0, frames.size(0) - 1, 1 / adjustment_factor).long()
        print(indices_to_interpolate)
        interpolated_frames = []
        for idx in indices_to_interpolate:
            t = idx % 1 
            interpolated_frame = interpolate_frames(frames[idx], frames[idx + 1], t)
            interpolated_frames.append(interpolated_frame)
       
        output_frames = torch.cat([frames, torch.stack(interpolated_frames)], dim=0)
        
    elif adjustment_factor < 1:
       
        indices_to_keep = torch.arange(0, frames.size(0), int(1 / adjustment_factor)).long()
        output_frames = frames[indices_to_keep]
    else:
        return video_tensor, info

    info['video_fps'] = target_fps

   

    return output_frames

if __name__ == "__main__":
    # Read a video file into a PyTorch tensor
    video_file = "/home/zottang/working_dir/Videos/video/test2.mov"
    video_tensor, audio, info = read_video(video_file)

    # Set the target fps (you can change this value to increase or decrease fps)
    target_fps = 30

    # Adjust fps and get the output video tensor
    print(video_tensor.shape, info)
    output_video = adjust_fps(video_tensor, info, target_fps)
    print(output_video.shape)
    # Write the output video tensor to a new file
    # output_file = "/home/zottang/working_dir/Videos/video/test2.mp4"
   
