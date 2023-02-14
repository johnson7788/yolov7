#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/1/29 15:45
# @File  : data_utils.py
# @Author:
# @Desc  : 一些数据操作的函数
import os
import yaml
import pandas as pd
import json
from tqdm import tqdm
import random
import collections
import shutil
from PIL import Image
from utils import read_data


def read_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
    return img

def generate_cosmetic_data():
    """
    生成化妆品的数据集，保存到特定目录，符合yolo的训练格式
    :param check_image: 检查图片是否能被Pillow打开
    """
    data_path = "../data/cosmetic"
    train_path = os.path.join(data_path, "train")
    dev_path = os.path.join(data_path, "dev")
    config_yaml = "../data/product_all.yaml"
    # 标签类别文件
    train_classes_path = os.path.join(train_path, "classes.txt")
    # 标注的目录
    train_annotation_dir = os.path.join(train_path, "labels")
    # 对应的原始图片的目录
    train_images_path = os.path.join(train_path, "images")
    # 其它说明文件
    train_notes_path = os.path.join(train_path, "notes.json")
    # 标签类别文件
    dev_classes_path = os.path.join(dev_path, "classes.txt")
    # 标注的目录
    dev_annotation_dir = os.path.join(dev_path, "labels")
    # 对应的原始图片的目录
    dev_images_path = os.path.join(dev_path, "images")
    print(f"删除旧的数据集")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    print(f"创建数据集目录")
    os.makedirs(train_images_path)
    os.makedirs(train_annotation_dir)
    os.makedirs(dev_images_path)
    os.makedirs(dev_annotation_dir)
    # 读取数据
    source_data = read_data()
    # 保存数据到指定目录
    # 保存到csv文件的列, gtbboxid,classid,imageid,lx,rx,ty,by,difficult,split, imagefilename, classfilename
    # 所有的商品
    image_size_counter = collections.Counter()
    categories = []
    class2id = {}
    for idx, one in enumerate(tqdm(source_data)):
        if idx % 200 == 0:
            print(f"完成了: {idx} 条")
        product = one["product"]
        if product in class2id:
            label_id = class2id[product]
        else:
            categories.append(product)
            label_id = len(class2id)
            class2id[product] = label_id
        imageid = one['md5']
        # bbox是左上角的点和右下角的点, 需要换成百分比格式
        x_min,y_min,x_max,y_max = one['bbox']
        split = random.choices(['train', 'dev'], [0.8, 0.2], k=1)[0]
        imagefilename_path = one["path"]
        if not os.path.exists(imagefilename_path):
            continue
        imagefilename = os.path.basename(imagefilename_path)
        # 检查imagefilename_path， 如果无法打开，那么就跳过这条数据
        try:
            img = read_image(image_path=imagefilename_path)
            # 图片的大小
            width, height = img.size
            image_size_counter[f"{width}x{height}"] += 1
        except Exception as e:
            print(f"图片: {imagefilename_path} 无法用Pillow打开，跳过这条数据")
            continue
        annotations = []
        center_x = (x_min + x_max) / 2 / width
        center_y = (y_min + y_max) / 2 / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        annotations.append([label_id, center_x, center_y, box_width, box_height])
        if split == "train":
            # 拷贝图片到训练目录
            src_filepath = os.path.join(train_images_path, imagefilename)
            shutil.copy(src=imagefilename_path,dst=src_filepath)
            label_txt = os.path.splitext(imagefilename)[0] + ".txt"
            label_txt_path = os.path.join(train_annotation_dir,label_txt)
        else:
            #拷贝到评估目录
            src_filepath = os.path.join(dev_images_path, imagefilename)
            shutil.copy(src=imagefilename_path,dst=src_filepath)
            label_txt = os.path.splitext(imagefilename)[0] + ".txt"
            label_txt_path = os.path.join(dev_annotation_dir,label_txt)
        with open(label_txt_path, "w") as f:
            for annotation in annotations:
                for idx, l in enumerate(annotation):
                    if idx == len(annotation) - 1:
                        f.write(f"{l}\n")
                    else:
                        f.write(f"{l} ")
    # 类别标签文件写入到文件
    with open(train_classes_path, 'w', encoding='utf8') as f:
        for c in categories:
            f.write(c + '\n')
    with open(dev_classes_path, 'w', encoding='utf8') as f:
        for c in categories:
            f.write(c + '\n')
    with open(train_notes_path, "w") as f:
        json.dump(class2id,f,ensure_ascii=False)
    print(f"生成comestic数据完成, 商品头图图片的尺寸统计: {image_size_counter}")
    # 生成配置文件
    config_data = {}
    config_data["download"] = "http://127.0.0.1/comestic.zip"
    config_data["train"] = train_path.replace("../","") # 改成data目录下
    config_data["val"] = dev_path.replace("../","")
    config_data["nc"] = len(categories)
    config_data["names"] = categories
    with open(config_yaml, "w") as f:
        yaml.dump(config_data, f, encoding="utf-8", allow_unicode=True)
    print(f"保存数据的配置到: {config_yaml}, 其中类别总数是: {len(categories)}")

if __name__ == '__main__':
    generate_cosmetic_data()
