import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from matplotlib import pyplot as plt
import random
import torchvision


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
    return [(os.path.join(root, img_name + '.jpg'), os.path.join(root, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class VideoImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, imgs_file, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = [i_id.strip() for i_id in open(imgs_file)]
        # self.imgs.sort()
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index].split(' ')
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        target = Image.open(os.path.join(self.root, gt_path)).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class VideoSequenceFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, gt_root, imgs_file, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.gt_root = gt_root
        self.imgs = [i_id.strip() for i_id in open(imgs_file)]
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_paths = self.imgs[index].split(',')
        img_list = []
        gt_list = []
        for img_path in img_paths:
            img = Image.open(os.path.join(self.root, img_path + '.jpg')).convert('RGB')
            target = Image.open(os.path.join(self.gt_root, img_path + '.png')).convert('L')
            img_list.append(img)
            gt_list.append(target)
        if self.joint_transform is not None:
            img_list, gt_list = self.joint_transform(img_list, gt_list)
        if self.transform is not None:
            imgs = []
            for img_s in img_list:
                imgs.append(self.transform(img_s).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
        if self.target_transform is not None:
            targets = []
            for target_s in gt_list:
                targets.append(self.target_transform(target_s).unsqueeze(0))
            targets = torch.cat(targets, dim=0)
        return imgs, targets

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    from torchvision import transforms

    import joint_transforms
    from torch.utils.data import DataLoader
    from config import msra10k_path, video_seq_path, video_seq_gt_path, video_train_path
    import numpy as np
    joint_transform = joint_transforms.Compose([
        joint_transforms.ImageResize(550),
        joint_transforms.RandomCrop(473),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomRotate(10)
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()

    # imgs_file = '/home/ty/data/video_saliency/train_all_DAFB2_DAVSOD_5f.txt'
    # train_set = VideoSequenceFolder(video_seq_path, video_seq_gt_path, imgs_file, joint_transform, img_transform, target_transform)
    imgs_file = '/home/ty/data/Pre-train/pretrain_all_seq_DUT_TR_DAFB2_DAVSOD.txt'
    train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=10, num_workers=12, shuffle=False)

    for i, data in enumerate(train_loader):
        input, target = data
        input = input.squeeze(0)
        target = target.squeeze(0)
        input = input.data.cpu().numpy()
        target = target.data.cpu().numpy()
        # np.savetxt('image.txt', input[0, 0, :, :])
        # input = input.transpose(0, 2, 3, 1)
        # target = target.transpose(0, 2, 3, 1)
        # # for i in range(0, input.shape[0]):
        # plt.subplot(2, 2, 1)
        # plt.imshow(input[0])
        # plt.subplot(2, 2, 2)
        # plt.imshow(target[0, :, :, 0])
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(input[1])
        # plt.subplot(2, 2, 4)
        # plt.imshow(target[1, :, :, 0])
        #
        # plt.show()
        print(input.shape)