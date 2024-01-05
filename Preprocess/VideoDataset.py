import os
import glob
import random
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoDataset(Dataset):
    def __init__(self, split, transform=None, ds_path="./Dataset"):

        if split == 'train':
            self.ds_path = os.path.join(ds_path, 'train')
        elif split == 'test':
            self.ds_path = os.path.join(ds_path, 'test')
        else:
            raise NotImplementedError

        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

            return img_source, img_target

    def __len__(self):
        return len(self.videos)