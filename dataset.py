import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import glob
import torch


class ImageFolder(data.Dataset):
    def __init__(self, root, mode='train', augmentation_prob=0.4, crop_size_min=300,
                 crop_size_max=500, data_num=0, gauss_size=21):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.crop_size_min = crop_size_min
        self.crop_size_max = crop_size_max
        self.data_num = data_num
        self.gauss_size = gauss_size

        self.data_dir_name = mode + '_img'
        self.label_dir_name = mode + '_label'

        
        self.data_paths = glob.glob(os.path.join(root, 'new_{}_set'.format(mode), self.data_dir_name, '*.png'))
        self.data_paths.sort()

        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        data_path = self.data_paths[index % len(self.data_paths)]
        image = Image.open(data_path)
        GT = Image.open(data_path.replace(self.data_dir_name, self.label_dir_name))

        Transform = []
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            if random.random() < 0.5:  # random rotation
                RotationDegree = random.randint(0, 3)
                RotationDegree = self.RotationDegree[RotationDegree]

                Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

                RotationRange = random.randint(-10, 10)
                Transform.append(T.RandomRotation((RotationRange, RotationRange)))

                Transform = T.Compose(Transform)
                image = Transform(image)
                GT = Transform(GT)

            if random.random() < 0.5:  # random crop
                crop_len = random.randint(self.crop_size_min, self.crop_size_max)
                i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_len, crop_len))
                image = F.crop(image, i, j, h, w)
                GT = F.crop(GT, i, j, h, w)

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = []

        Transform.append(T.Resize((512, 512)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        GT = Transform(GT)
    
        Norm_ = T.Normalize((0.5,), (0.5,))
        image = Norm_(image)

        loss_weight = torch.ones_like(GT)
        if self.gauss_size > 0:
            gauss_trans = T.GaussianBlur(kernel_size=self.gauss_size, sigma=(5.0, 5.0))
            weight_tmp = 1.0 - gauss_trans(GT)
            loss_weight = 0.5*loss_weight + 0.5*weight_tmp

        return image, GT, loss_weight

    def __len__(self):
        """Returns the total number of font files."""
        if self.data_num > 0:
            return self.data_num
        else:
            return len(self.data_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    root = '/Users/chenzhengyang/gitRepo/Image_Segmentation/data'
    dataset = ImageFolder(root, mode='train', augmentation_prob=1.0, gauss_size=21)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=5,
                                  shuffle=False,
                                  num_workers=1,
                                )
    # data, label, loss_weight = dataset[0]
    # trans = T.ToPILImage()
    # image = trans(loss_weight)
    #
    # image.show()


    import torchvision
    for data, label, loss_weight in data_loader:
        print(label.shape)
        # torchvision.utils.save_image(label.float()[0], 'tmp/tmp.png')
        break