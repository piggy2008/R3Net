import os
import os.path

import torch.utils.data as data
from PIL import Image
from matplotlib import pyplot as plt



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

if __name__ == '__main__':
    from torchvision import transforms
    import joint_transforms
    from torch.utils.data import DataLoader
    from config import msra10k_path, video_train_path

    joint_transform = joint_transforms.Compose([
        joint_transforms.RandomCrop(473),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomRotate(10)
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()

    imgs_file = '/home/qub/data/saliency/Pre-train/pretrain_all_seq3.txt'
    train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=14, num_workers=12, shuffle=True)

    for i, data in enumerate(train_loader):
        input, target = data
        print(input.size())
