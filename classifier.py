import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SCALE = 224


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



model = SimpleVGGClassifier(IMAGE_SCALE).cuda()
model.load_state_dict(torch.load("data/model3.pt"))
model.eval()


def classify_pizza(image: np.ndarray) -> bool:
    image = image[..., ::-1].copy()
    resized_img = cv2.resize(image, dsize=(IMAGE_SCALE, IMAGE_SCALE), interpolation=cv2.INTER_CUBIC)
    scaled_img = np.transpose(resized_img, [2, 0, 1]) / np.max(resized_img)
    scaled_img = torch.tensor(scaled_img).type(torch.cuda.FloatTensor)
    result = model(scaled_img.unsqueeze(dim=0))
    result = torch.flatten(result)
    return result[1] > result[0]

