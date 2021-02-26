from os import walk
import os

import cv2

_, _, filenames = next(walk('data/dataset'))
# _, _, filenames = next(walk('data/train/black_side'))

for file in filenames:
    img_path = os.path.join('data', 'dataset', file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (600, 600))
    cv2.imshow('ImageWindow', img)
    cv2.waitKey(0)
