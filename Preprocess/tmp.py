import os
from os import path
import subprocess

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args['preprocessed_root'], dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    print(command)
    subprocess.call(command, shell=True)

args = {'preprocessed_root': './'}
video_file = '/home/amanankesh/working_dir/FCTFG/Preprocess/test.mov'

process_audio_file(video_file, args)
