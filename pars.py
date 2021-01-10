import os

import numpy as np
import urllib.request
import cv2

NAMES = [
    "Pizza in a box",
    "Pizza top view",
    "The bottom of the pizza top view",
    "Pizza bottom view at an angle",
    "Pizza sides view at an angle of 90",
    "Pizza sides view at an angle",
    "Cheque",
    "Pizza, view from a small angle"
]


class Parser:
    """
    Класс для работы с информцией о дефектах
    """

    def __init__(self, path_to_csv='data/controlling_ml.csv'):
        """
        Конструктор, распаковывает данные из файла
        :param path_to_csv: Путь к файлу
        """
        with open(path_to_csv, encoding="utf8") as csv_file:
            lines = csv_file.readlines()

        self.data = [row.replace('\n', '').split(',') for row in lines[1:]]
        self.DEFECTS_NAMES = lines[0].replace('\n', '').split(',')[10:]
        self.NUMBER_OF_COLUMNS = len(self.data[0])
        self.FIRST_DEFECT = self.data[0].index('0')  # TODO: ну ты сам поял, ты там давай, не умирай

    def show_by_defects(self, defects_list=[], pic_nums=[]):
        for row in self.data:
            defects = row[self.FIRST_DEFECT:]
            comp_array = ['0'] * (self.NUMBER_OF_COLUMNS - self.FIRST_DEFECT)
            for defect in defects_list:
                comp_array[defect - self.FIRST_DEFECT] = '1'

            if defects == comp_array:
                for pic_num in pic_nums:
                    try:
                        img = self.url_to_image(row[2 + pic_num])
                        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                        cv2.imshow(NAMES[pic_num], img)
                    except:
                        print(f'fail for {row[2 + pic_num]}')
                cv2.waitKey(0)

    def save_by_defects(self, folder, pic_num, defects_list=[], start=0, end=1000):
        """
        Сохранение изображений
        :param folder: Название папки
        :param defects_list: Список присутствующий дефектов
        :param pic_num: Номер фотографии
        :param start: Ограничение по количеству фотографий
        :param end: Ограничение по количеству фотографий
        :return:
        """
        counter = -1
        for row in self.data:
            defects = row[self.FIRST_DEFECT:]
            comp_array = ['0'] * (self.NUMBER_OF_COLUMNS - self.FIRST_DEFECT)
            for defect in defects_list:
                comp_array[defect - self.FIRST_DEFECT] = '1'

            path_to_folder = os.path.join('data', folder)

            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)

            if defects == comp_array:
                counter += 1
                if counter < start or end <= counter:
                    continue
                try:
                    url = row[2 + pic_num]
                    img = self.url_to_image(url)
                    im_name = os.path.join(path_to_folder, url[url.rfind('/') + 1:])
                    cv2.imwrite(im_name, img)
                except:
                    print(f'fail for {url}')


    @staticmethod
    def url_to_image(url):
        """
        download the image, convert it to a NumPy array, and then read
        it into OpenCV format
        :param url: Url-addres
        :return:
        """
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image


if __name__ == '__main__':
    parser = Parser('data/controlling_ml.csv')
    parser.save_by_defects(folder='test1', defects_list=[15], pic_num=2, start=0, end=5)  # pic_nums=[2, 3],
    parser.save_by_defects(folder='test2', defects_list=[15], pic_num=2, start=5, end=10)  # pic_nums=[2, 3],
    # parser.save_by_defects(folder='test/white_bottom', defects_list=[15], pic_num=2, start=200)  # pic_nums=[2, 3],
    # parser.save_by_defects(folder='black_bottom', defects_list=[17], pic_num=2)  # pic_nums=[2, 3],
    # parser.save_by_defects(folder='white_side', defects_list=[14], pic_num=2)  # pic_nums=[2, 3],
    # parser.save_by_defects(folder='black_side', defects_list=[16], pic_num=2)  # pic_nums=[2, 3],
    # parser.save_by_defects(folder='test/normal', defects_list=[], pic_num=2, limit=200)  # pic_nums=[2, 3],
