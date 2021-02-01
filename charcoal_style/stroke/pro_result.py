#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-11 下午3:40
# @Author  : Jehovah
# @File    : pro_result.py
# @Software: PyCharm

import os
import cv2
import os
import PIL.Image as Image
from PIL import ImageDraw
import numpy as np


def cropImage():
    path = '/home/jehovah/PycharmProject/Image_enhance/sys_multihead_multi_2'
    # path = '/home/jehovah/PycharmProject/wnn_data/base'
    # path = '/home/jehovah/PycharmProject/B2A/base'
    savepath = path + '_200'

    new_width = 200
    new_height = 250
    i = 1
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for root, _, files in os.walk(path):
        for file in sorted(files):
            image_root = os.path.join(root, file)
            im = Image.open(image_root)
            width, height = im.size  # Get dimensions

            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            im = im.crop((left, top, right, bottom))
            # im.save(savepath + '/' + str(i)+'.jpg', quality=100)
            # im.save(savepath + '/' + str(int(file.split('_')[0]) + 1) + '.jpg', quality=100)
            im.save(savepath + '/' + file, quality=100)
            i = i + 1

def catImage7():
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix_parsing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/sketch',
    # ]
    coofoc = [
        "/home/meimei/mayme/code/sketch_parsing_m3/results/spade_celeb/spade_celeb_rename/10552.jpg", ###spade
        "/home/meimei/mayme/code/sketch_parsing_m3/results/spade_celeb/spade_style_cerename/10552.jpg", ###spade_style
        "/home/meimei/mayme/code/sketch_parsing_m3/results/spade_celeb/spade_100m2_cerename/10552.jpg", ###100m2
        "/home/meimei/mayme/code/sketch_parsing_m3/results/stroke50l1__celeb/stroke7_2e-3_celeb_rename/10552.jpg", ###stroke7
        "/home/meimei/mayme/code/sketch_parsing_m3/results/stroke50l1__celeb/stroke7_2e-3_style_celeb_rename/10552.jpg", ###stroke7_style
        "/home/meimei/mayme/code/sketch_parsing_m3/results/stroke50l1__celeb/stroke7_20m2style_celeb_rename/10552.jpg", ###stroke7_20m2_style
        "/home/meimei/mayme/data/128testA_resize/10552.jpg"
    ]
    DIR = coofoc[0]
    savepath = '/home/meimei/mayme/data/celeb_cat'
    # len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 512  # 250
    UNIT_WIGHT = 512  # 200
    w = 8
    TARGET_WIDTH = (UNIT_WIGHT + w) * len(coofoc)
    TARGET_WIDTH = int(TARGET_WIDTH)
    # #1
    # x1 = 210
    # y1 = 100
    # x2 = 220
    # y2 = 340
    # 2
    # l_eye
    # x1 = 160
    # y1 = 210
    # x2 = 220
    # y2 = 340
    # #3
    # x1 = 210
    # y1 = 100
    # x2 = 160
    # y2 = 210
    # 4 眼睛、胡子
    # x1 = 160
    # y1= 210
    # x2 = 220
    # y2 = 290
    # x1 = 180
    # y1= 50
    # x2 = 160
    # y2 = 210
    # 5 眼睛、衣服
    # x1 = 160
    # y1 = 210
    # x2 = 400
    # y2 = 400
    x1 = 150
    y1= 200
    x2 = 280
    y2 = 200
    lenth = 100
    lenth1 = 100
    # name_list = os.listdir(coofoc[0])
    for i in range(0, 1):
        print(i)
        pipbe = []
        pip11be = Image.open(coofoc[0]).convert('RGB')
        pip12be = Image.open(coofoc[1]).convert('RGB')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        for k in range(2, len(coofoc)):
            pip13be = Image.open(coofoc[k]).convert('RGB')
            pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, (UNIT_SIZE + 252 + 8)), color='#F5F5F5')
        for k in range(0, len(pipbe)):
            target.paste(pipbe[k], ((UNIT_WIGHT + w) * k, 0, (UNIT_WIGHT + w) * k + UNIT_WIGHT, UNIT_SIZE))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x1, y1), (k * (UNIT_WIGHT + w) + x1 + lenth, y1),
                       (k * (UNIT_WIGHT + w) + x1 + lenth, y1 + lenth), (k * (UNIT_WIGHT + w) + x1, y1 + lenth),
                       (k * (UNIT_WIGHT + w) + x1, y1)], width=4,
                      fill="red")

            img_detail1 = pipbe[k].crop((x1, y1, x1 + lenth, y1 + lenth))
            # print(k * UNIT_WIGHT + x1, y1, k* UNIT_WIGHT + x1 + lenth, y1 + lenth)
            img_detail1 = img_detail1.resize((252, 252))
            target.paste(img_detail1,
                         (k * (UNIT_WIGHT + w), UNIT_SIZE + 8, k * (UNIT_WIGHT + w) + 252, UNIT_SIZE + 252 + 8))
            # print(np.array(img_detail1))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x2, y2), (k * (UNIT_WIGHT + w) + x2 + lenth1, y2),
                       (k * (UNIT_WIGHT + w) + x2 + lenth1, y2 + lenth1), (k * (UNIT_WIGHT + w) + x2, y2 + lenth1),
                       (k * (UNIT_WIGHT + w) + x2, y2)], width=4,
                      fill="green")
            img_detail2 = pipbe[k].crop((x2, y2, x2 + lenth1, y2 + lenth1))
            img_detail2 = img_detail2.resize((252, 252))
            target.paste(img_detail2,
                         (k * (UNIT_WIGHT + w) + 8 + 252, 512 + 8, k * (UNIT_WIGHT + w) + 8 + 2 * 252, 512 + 252 + 8))

        target.save(savepath+'/10552.jpg' , quality=100)


def catImages7():
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix_parsing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/sketch',
    # ]
    coofoc = [
        "/home/meimei/mayme/code/sketch_parsing_m3/results/spade_test/spade_test_rename/", ###spade
        "/home/meimei/mayme/code/sketch_parsing_m3/results/spade_test/spade_1e5_test_rename/", ###spade_style
        "/home/meimei/mayme/code/sketch_parsing_m3/results/spade_test/spade_100m2_test_rename/", ###100m2
        "/home/meimei/mayme/code/sketch_parsing_m3/results/stroke50l1__test/stroke7_2e-3_test_rename/", ###stroke7
        "/home/meimei/mayme/code/sketch_parsing_m3/results/stroke50l1__test/stroke7_2e-3_style_test_rename/", ###stroke7_style
        "/home/meimei/mayme/code/sketch_parsing_m3/results/stroke50l1__test/stroke7_20m2style_test_rename/", ###stroke7_20m2_style
    ]
    DIR = coofoc[0]
    savepath = '/home/meimei/mayme/data/test_cat'
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 512  # 250
    UNIT_WIGHT = 512  # 200
    w = 8
    TARGET_WIDTH = (UNIT_WIGHT + w) * len(coofoc)
    TARGET_WIDTH = int(TARGET_WIDTH)
    # #1
    # x1 = 210
    # y1 = 100
    # x2 = 220
    # y2 = 340
    # 2
    # l_eye
    # x1 = 160
    # y1 = 210
    # x2 = 220
    # y2 = 340
    # #3
    # x1 = 210
    # y1 = 100
    # x2 = 160
    # y2 = 210
    # 4 眼睛、胡子
    # x1 = 160
    # y1= 210
    # x2 = 220
    # y2 = 290
    # x1 = 180
    # y1= 50
    # x2 = 160
    # y2 = 210
    # 5 眼睛、衣服
    # x1 = 160
    # y1 = 210
    # x2 = 400
    # y2 = 400
    x1 = 160
    y1= 210
    x2 = 240
    y2 = 450
    lenth = 70
    name_list = os.listdir(coofoc[0])
    for i in range(0, len_dir):
        print(i)
        pipbe = []
        pip11be = Image.open(coofoc[0] + '/'+ name_list[i]).convert('RGB')
        pip12be = Image.open(coofoc[1] + '/'+ name_list[i]).convert('RGB')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        for k in range(2, len(coofoc)):
            pip13be = Image.open(coofoc[k] + '/'+ name_list[i]).convert('RGB')
            pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, (UNIT_SIZE + 252 + 8)), color='#F5F5F5')
        for k in range(0, len(pipbe)):
            target.paste(pipbe[k], ((UNIT_WIGHT + w) * k, 0, (UNIT_WIGHT + w) * k + UNIT_WIGHT, UNIT_SIZE))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x1, y1), (k * (UNIT_WIGHT + w) + x1 + lenth, y1),
                       (k * (UNIT_WIGHT + w) + x1 + lenth, y1 + lenth), (k * (UNIT_WIGHT + w) + x1, y1 + lenth),
                       (k * (UNIT_WIGHT + w) + x1, y1)], width=4,
                      fill="red")

            img_detail1 = pipbe[k].crop((x1, y1, x1 + lenth, y1 + lenth))
            # print(k * UNIT_WIGHT + x1, y1, k* UNIT_WIGHT + x1 + lenth, y1 + lenth)
            img_detail1 = img_detail1.resize((252, 252))
            target.paste(img_detail1,
                         (k * (UNIT_WIGHT + w), UNIT_SIZE + 8, k * (UNIT_WIGHT + w) + 252, UNIT_SIZE + 252 + 8))
            # print(np.array(img_detail1))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x2, y2), (k * (UNIT_WIGHT + w) + x2 + lenth, y2),
                       (k * (UNIT_WIGHT + w) + x2 + lenth, y2 + lenth), (k * (UNIT_WIGHT + w) + x2, y2 + lenth),
                       (k * (UNIT_WIGHT + w) + x2, y2)], width=4,
                      fill="green")
            img_detail2 = pipbe[k].crop((x2, y2, x2 + lenth, y2 + lenth))
            img_detail2 = img_detail2.resize((252, 252))
            target.paste(img_detail2,
                         (k * (UNIT_WIGHT + w) + 8 + 252, 512 + 8, k * (UNIT_WIGHT + w) + 8 + 2 * 252, 512 + 252 + 8))

        target.save(savepath + '/'+ name_list[i], quality=100)


if __name__ == '__main__':
    catImage7()
    # cropImage()
