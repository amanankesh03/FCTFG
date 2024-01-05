import torch
from torchvision.io import read_video
from PIL import Image
import os

def video_to_images(video_path, output_folder):
    # Read the video
    video, audio, info = read_video(video_path, pts_unit='sec')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract frames and save as images
    frame_number = 0
    for frame in video:
        # Convert frame to PIL image
        print(frame.shape)
        image = Image.fromarray(frame.numpy().astype('uint8')).resize((512, 512))
        image_path = os.path.join(output_folder, f"frame_{frame_number:04d}.png")

        # Save the PIL image
        image.save(image_path)

        frame_number += 1

    print(f"Conversion complete. {frame_number} frames saved to {output_folder}")


if __name__ == "__main__":
        
    # Example usage
    video_folder_path = '/home/amanankesh/working_dir/FCTFG/Preprocess/video_data/'
    output_folder = '/home/amanankesh/working_dir/FCTFG/DATA/video/'

    video_list = os.listdir(video_folder_path)
    print(video_list)
    video_path = os.path.join(video_folder_path, video_list[0])
    output_path = os.path.join(output_folder, video_list[0])
    video_to_images(video_path, output_path)