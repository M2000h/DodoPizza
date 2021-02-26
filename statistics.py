import numpy as np
import pars


def get_defects():
    with open('data/controlling_ml.csv', encoding="utf8") as file:
        lines = file.readlines()
        array = [row.split(',') for row in lines]

    defects = np.zeros(35, int)
    for row in array:
        row[44] = row[44][:-1]
        deff = row[10:]
        for i in range(len(deff)):
            if deff[i] == '1':
                defects[i] += 1
    set_size = len(array) - 1
    names = array[0][10:]
    for i in range(35):
        print(f'{i + 10}) {names[i]}, {defects[i]}')
    print()
    print('Размер выборки:', set_size)


def get_few_defects(defects_list=[]):
    parser = pars.Parser('data/controlling_ml.csv')
    counter = 0
    for row in parser.data:
        defects = row[parser.FIRST_DEFECT:]
        comp_array = ['0'] * (parser.NUMBER_OF_COLUMNS - parser.FIRST_DEFECT)
        for defect in defects_list:
            comp_array[defect - parser.FIRST_DEFECT] = '1'
        if defects == comp_array:
            counter += 1

    print('Деффекты:')
    for i in defects_list:
        print(f'{i}) {parser.DEFECTS_NAMES[i - 10]}')
    print(counter)


get_few_defects([22])

# get_defects()
