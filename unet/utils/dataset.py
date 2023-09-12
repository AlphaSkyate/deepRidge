from abc import ABC

import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import json
import numpy as np
from PIL import Image


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.bmp'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


class DataPre(object):
    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.bmp'))

    def split(self, width):
        for img_file_path in self.imgs_path:
            img_o = cv2.imread(img_file_path)
            width_o = img_o.shape[1]
            height_o = img_o.shape[0]
            print("===================================")
            print(img_file_path)
            if width_o > width:
                parts_num = int(width_o / width)
                print("parts:", parts_num)
                if parts_num > 1:
                    for parts_index in range(0, parts_num):
                        save_path = os.path.join(self.target_path,
                                                 os.path.splitext(os.path.basename(img_file_path))[-2] + str(
                                                     parts_index) + ".bmp")
                        # print(save_path)
                        img_part = img_o[0:(height_o - 1), (width * parts_index):(width * (parts_index + 1))]
                        cv2.imwrite(save_path, img_part)


def a2i(indata):
    mg = Image.new('L', indata.transpose().shape)
    mn = indata.min()
    a = indata - mn
    mx = a.max()
    a = a * (255. / mx)
    mg.putdata(a.ravel())
    return mg


class MaskGen(object):
    def __init__(self, json_path, data_path, out_path):
        self.json_path = json_path
        self.data_path = data_path
        self.out_path = out_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.bmp'))
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            self.data_label = data["_via_img_metadata"]

    def generator(self):
        for file_element in self.data_label:
            filename = self.data_label[file_element]["filename"].split('/')[-1]
            img_file_path = os.path.join(self.data_path, filename)
            mask_save_path = os.path.join(self.out_path, filename)
            print(img_file_path)
            print(mask_save_path)
            img_o = cv2.imread(img_file_path)
            width_o = img_o.shape[1]
            height_o = img_o.shape[0]
            img_array = np.zeros((height_o, width_o))
            region_list = self.data_label[file_element]["regions"]
            for region_group in region_list:
                region = region_group["shape_attributes"]
                region_x = region["all_points_x"]
                region_y = region["all_points_y"]
                pre_x = 0
                pre_y = 0
                for index in range(0, len(region_x)):
                    if index == 0:
                        pre_y = region_y[index]
                        pre_x = region_x[index]
                    else:
                        cur_y = region_y[index]
                        cur_x = region_x[index]

                        width_line = cur_x-pre_x
                        height_line = cur_y-pre_y
                        point_num = max(abs(width_line), abs(height_line))
                        if point_num != 0:
                            step_x = width_line/point_num
                            step_y = height_line/point_num
                            for cp in range(0, point_num):
                                point_x = int(pre_x + cp * step_x)
                                point_y = int(pre_y + cp * step_y)
                                if (point_x >= 0) & (point_x < width_o) & (point_y >= 0) & (point_y < height_o):
                                    img_array[int(pre_y + cp * step_y)][int(pre_x + cp * step_x)] = 255
                                    if (point_x >= 0) & (point_x < width_o - 1) & (point_y >= 0) & (point_y < height_o - 1):
                                        img_array[int(pre_y + cp * step_y) + 1][int(pre_x + cp * step_x)] = 255

                        pre_y = cur_y
                        pre_x = cur_x

            # mask_img = a2i(img_array)
            cv2.imwrite(mask_save_path, img_array)
            # mask_img.save(mask_save_path)


if __name__ == "__main__":
    # isbi_dataset = ISBI_Loader("data/train/")
    # print("数据个数：", len(isbi_dataset))
    # train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
    #                                            batch_size=2,
    #                                            shuffle=True)
    # for image, label in train_loader:
    #     print(image.shape)

    # dataset_l = DataPre("../data/data_noise/00", "../data/data_noise/res")
    # dataset_l.split(416)

    mg = MaskGen("../../data/via/data_label_task/task9/task9.json", "../../data/via/data_label_task/task9", "../../data/via/data_label_task/task9_mask")
    mg.generator()
