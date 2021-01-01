import numpy as np
import urllib.request

import cv2

NAMES = [
    "Pizza in a box",
    "Pizza top view",
    "The bottom of the pizza view from above",
    "Pizza bottom view at an angle",
    "Pizza sides view at an angle of 90",
    "Pizza sides view at an angle of 90",
    "Cheque",
    "Pizza, view from a small angle"
]


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


with open('controlling_ml.csv', encoding="utf8") as file:
    lines = file.readlines()
    array = [row.split(',') for row in lines]
    for i in range(len(array[0])):
        print(i, array[0][i])
    for row in array[1:]:
        for pic_num in [1, 2, 3, 4, 5, 7]:
            img = url_to_image(row[2 + pic_num])
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
            cv2.imshow(NAMES[pic_num], img)
        cv2.waitKey(0)
