import pydicom
import numpy as np
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from model import Model
import os
from pydicom.data import get_testdata_files

# filename = "data/09115862_ANO_20190805_1451152_114359/SeriesDescription_13/00000007.dcm"
# ds = pydicom.dcmread(filename)
# print(ds.PatientName)
# pic = plt.imshow(ds.pixel_array, "gray")
# print(ds.pixel_array.shape)
# plt.show()

series_dict = {}
titles = []
title_dict = {}


def read():
    global series_dict
    global titles
    global title_dict

    file = open("20190805.csv", 'r')
    csv_reader = csv.reader(file)
    is_title = True
    for row in csv_reader:
        if is_title is True:
            titles = row
            is_title = False
            continue

        series_dict[row[1]] = row

    count = 0
    for t in titles:
        title_dict[t] = count
        count += 1


def find_data(number):
    global series_dict
    if number in series_dict:
        return series_dict[number]

    return None


def find_folder(number):
    number = number[2:]
    for parent in os.listdir("/Volumes/Seagate Exp/HCC_MRI"):
        for dirname in os.listdir("/Volumes/Seagate Exp/HCC_MRI/" + parent):
            if dirname.split('_')[0] == number:
                return "/Volumes/Seagate Exp/HCC_MRI/" + parent + "/" + dirname


def get_series_in_folder(number, series_no):
    dir_path = find_folder(number) + "/SeriesDescription_" + series_no
    return dir_path, len(os.listdir(dir_path))


def tag_translate(label):
    if label == 'HCC':
        return [1, 0, 0, 0, 0]
    elif label == 'ICC':
        return [0, 1, 0, 0, 0]
    elif label == 'MT':
        return [0, 0, 1, 0, 0]
    elif label == 'META':
        return [0, 0, 0, 1, 0]
    elif label == 'BEN':
        return [0, 0, 0, 0, 1]
    return [0, 0, 0, 0, 0]


def init_series_data(number):
    all_length = 0
    file_list = []

    path, length = get_series_in_folder(number, series_dict[number][title_dict['series_1']])
    all_length += length
    for file in os.listdir(path):
        ds = pydicom.dcmread(path + "/" + file)
        file_list.append(np.array(ds.pixel_array).reshape((1, -1)).tolist()[0])

    path, length = get_series_in_folder(number, series_dict[number][title_dict['series_2']])
    all_length += length
    for file in os.listdir(path):
        ds = pydicom.dcmread(path + "/" + file)
        file_list.append(np.array(ds.pixel_array).reshape((1, -1)).tolist()[0])

    path, length = get_series_in_folder(number, series_dict[number][title_dict['series_3']])
    all_length += length
    for file in os.listdir(path):
        ds = pydicom.dcmread(path + "/" + file)
        file_list.append(np.array(ds.pixel_array).reshape((1, -1)).tolist()[0])

    # path, length = get_series_in_folder(series_dict[data][title_dict['No']],
    #                                     series_dict[data][title_dict['series_4']])
    # all_length += length
    # file_list = []
    # for file in os.listdir(path):
    #     ds = pydicom.dcmread(path + "/" + file)
    #     file_list.append(np.array(ds.pixel_array).reshape((1, -1)))
    # series_data.append(file_list)

    return all_length, np.array(file_list), tag_translate(series_dict[number][title_dict['label']])


if __name__ == "__main__":
    read()
    print("reading finished.")
    print(get_series_in_folder('ZS18140790', "10"))
    model = Model(0.001, 1, 5)
    length, input_arr, tag = init_series_data('ZS18140790')
    input_arr = np.pad(input_arr, ((0, 300 - input_arr.shape[0]), (0, 0)), 'constant')
    seq_len_arr = []
    for i in input_arr.tolist():
        seq_len_arr.append(len(i))

    seq_len_arr = np.array(seq_len_arr)
    model.train(1, [input_arr], seq_len_arr, [tag])
