import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from imgaug import augmenters as iaa
import imgaug as ia
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

DATA_PATH = '/data'
train_path = os.path.join(DATA_PATH, 'train')
val_path = os.path.join(DATA_PATH, 'test')

NUM_EPOCHS = 100
IMAGE_SHAPE = 224
BATCH_SIZE = 12

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class TrainAugTransform:
    def __init__(self, rotation):
        self.aug = iaa.Sequential([
            iaa.Scale((IMAGE_SHAPE, IMAGE_SHAPE)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=rotation, mode='symmetric'),
            iaa.Sometimes(0.2, iaa.Dropout(p=(0, 0.1))),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = np.transpose(img, [2, 0, 1]) / np.max(img)
        return img


class TestAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Scale((IMAGE_SHAPE, IMAGE_SHAPE)),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = np.transpose(img, [2, 0, 1]) / np.max(img)
        return img


human_dataset1 = ImageFolder(train_path, transform=TrainAugTransform(-12))
human_dataset2 = ImageFolder(train_path, transform=TrainAugTransform(0))
human_dataset3 = ImageFolder(train_path, transform=TrainAugTransform(12))
human_dataset = ConcatDataset([human_dataset1, human_dataset2, human_dataset3])

data_loader = DataLoader(human_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ImageFolder(val_path, transform=TestAugTransform()), batch_size=BATCH_SIZE, shuffle=True)

import torch.nn.functional as F


class SimpleVGGClassifier(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.kernel_size = 3
        self.conv1 = self.conv_block(3, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = self.conv_block(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = self.conv_block(128, 256, 4)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = self.conv_block(256, 512, 4)
        self.bn4 = nn.BatchNorm2d(512)
        # n x 512 x 14 x 14

        self.linear1 = nn.Linear(512 * 14 * 14, 1000)
        self.lrelu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(1000, 1000)
        self.lrelu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(p=0.3)
        self.linear3 = nn.Linear(1000, 2)

    def conv_block(self, start_volume, end_volume, num_conv=1) -> nn.Module:
        block = nn.Sequential()
        for i in range(num_conv):
            name_conv, name_relu = 'conv_{}_{}'.format(i, start_volume), 'relu_{}'.format(i)
            block.add_module(name_conv, nn.Conv2d(start_volume if i == 0 else end_volume,
                                                  end_volume,
                                                  self.kernel_size,
                                                  padding=1))
            block.add_module(name_relu, nn.ReLU())
        name_pool = 'pool_{}'.format(start_volume)
        block.add_module(name_pool, nn.MaxPool2d(kernel_size=2))
        return block

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.linear1(x.view(input.shape[0], -1))
        x = self.lrelu1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.lrelu2(x)
        x = self.drop2(x)
        x = self.linear3(x)
        return F.softmax(x, dim=1)

from torch.optim import Adagrad

model = SimpleVGGClassifier(IMAGE_SHAPE).cuda() if use_cuda else SimpleVGGClassifier(IMAGE_SHAPE)
opt = Adagrad(model.parameters(), lr=0.001)
