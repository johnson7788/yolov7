#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/20 2:20 下午
# @File  : yolo_api.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 推理和训练的api
######################################################
# 包括训练接口api和预测接口api
# /api/train
# /api/predict
######################################################

import logging
import re
import sys

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

import os
import time
from pathlib import Path
import requests
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


from flask import Flask, request, jsonify, abort

app = Flask(__name__)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class YOLOModel(object):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.label_list = ['A1', 'A2', 'A3', 'A4','A5' ,'A6' ,'A7' ,'B1' ,'B2' ,'B3', 'B4','B5' ,'B6' ,'B7', 'C1', 'Keyboard']
        self.label_list_cn = ['A1', 'A2', 'A3', 'A4','A5' ,'A6' ,'A7' ,'B1' ,'B2' ,'B3', 'B4','B5' ,'B6' ,'B7', 'C1', 'Keyboard']
        #给每个类别的候选框设置一个颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.label_list]
        self.num_labels = len(self.label_list)
        # 判断使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # 预测的batch_size大小
        self.train_batch_size = 8
        # 预测的batch_size大小
        self.predict_batch_size = 16
        #模型的名称或路径
        self.weights = 'runs/train/gy/weights/best.pt'      #不用last.pt # 'yolov5s.pt'
        self.source = 'images_dir'  #图片目录
        self.img_size = 864   #像素
        self.conf_thres = 0.5  #置信度, 大于这个置信度的才类别才取出来
        self.iou_thres = 0.45  #IOU的NMS阈值
        self.view_img = False   #是否显示图片的结果
        self.save_img = True    #保存图片预测结果
        self.save_conf = False  #同时保存置信度到save_txt文本中
        self.classes = None  # 0, 1, 2 ，只过滤出我们希望的类别, None表示保留所有类别
        self.agnostic_nms = False #使用nms算法
        self.project = 'runs/api' #项目保存的路径
        self.image_dir = os.path.join(self.project, 'images')   #保存从网络下载的图片
        self.predict_dir = os.path.join(self.project, 'predict')   #保存预测结果
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
        self.load_predict_model()

    def load_train_model(self):
        """
        初始化训练的模型
        :return:
        """
        pass
        logger.info(f"训练模型{self.tuned_checkpoint_S}加载完成")

    def load_predict_model(self):
        # Load model
        self.predict_model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.predict_model.stride.max())  # model stride
        logger.info(f"预测模型{self.weights}加载完成")

    def download_file(sefl, url, save_dir):
        """
        我们返回绝对路径
        :param url: eg: http://127.0.0.1:9090/2007.158710001-01.jpg
        :param save_dir: eg: /tmp/
        :return:  /tmp/2007.158710001-01.jpg
        """
        local_filename = url.split('/')[-1]
        save_dir_abs = Path(save_dir).absolute()
        save_file = os.path.join(save_dir_abs, local_filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return save_file

    def detect(self, data):
        """
        返回的bboxes是实际的坐标，x1，y1，x2，y2，是左上角和右下角的坐标
        :param data: 图片数据的列表 [image1, image2]
        :return: [[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...] bboxes是所有的bboxes, confidence是置信度， labels是所有的bboxes对应的label，
        """
        #检查设置的图片的大小和模型的步长是否能整除
        imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
        #下载数据集, images保存图片的本地的路径
        images = []
        for url in data:
            if url.startswith('http'):
                image = self.download_file(url, self.image_dir)
            else:
                #不是http的格式的，是本地文件的，那么直接使用即可
                image = url
            images.append(image)
        #设置数据集
        dataset = LoadImages(path=self.image_dir, img_size=imgsz, stride=self.stride)
        # 这里我们重设下images，我们只要自己需要的images既可, dataset.nf, 即number_files, 文件数量也需要修改下
        dataset.files = images
        dataset.nf = len(images)
        dataset.video_flag = [False] * len(images)
        # 设置模型
        predict_model = self.predict_model
        # 运行推理
        if self.device.type != 'cpu':
            predict_model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(predict_model.parameters())))  # run once
        #计算耗时
        t0 = time.time()
        # path是图片的路径，img是图片的改变size后的numpy格式[channel, new_height,new_witdh], im0s是原始的图片,[height, width,channel], eg: (2200, 1700, 3), vid_cap 是None如果是图片，只对视频有作用
        results = []
        for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            # 如果是GPU，会放到GPU上
            img = torch.from_numpy(img).to(self.device)
            #转换成float
            img = img.float()
            #归一化
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #扩展一个batch_size维度, [batch_isze, channel, new_height, new_witdh], eg:  torch.Size([1, 3, 640, 512])
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            #开始推理, time_synchronized是GPU的同步
            t1 = time_synchronized()
            # pred模型的预测结果 [batch_size,hidden_size, other] eg: torch.Size([1, 20160, 8]), 8代表 (x1, y1, x2, y2, conf, cls1, cls2, cls3...), 前4个是bbox坐标，conf是置信度，cls是类别的，cls1代表是类别1的概率
            pred = predict_model(img, augment=False)[0]

            #使用 NMS, pred 是一个列表，里面是,一个元素代表一张图片, 一个元素的维度是 [bbox_num, other], other代表(x1, y1, x2, y2, conf, max_cls_prob) eg: torch.Size([5, 6])
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # 处理 detections,i是索引，det是所有的bbox, torch.Size([3, 6])，代表3个bbox，6代表 (x1, y1, x2, y2, 置信度, 类别id)
            for i, det in enumerate(pred):  # detections per image
                # s 是初始化一个空字符串，用于打印预测结果，im0是原始图片, frame是对于视频而言的, 这里pred一定是以个元素，因为我们迭代的是一张图片
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                #p原始图片的绝对路径
                p = Path(p)  # to Path
                save_path = os.path.join(self.predict_dir, p.name)  #预测后的保存的图片的路径
                s += ' 预测的图片尺寸高度宽度%gx%g，' % img.shape[2:] # print string, eg '640x480 '
                s += '原始尺寸为高度宽度%sx%s，' % im0.shape[:2]  # print string, eg '640x480 '
                # 图片的width,height, width, height, eg: tensor([1700, 2200, 1700, 2200]), 用于下面的归一化
                #如果det不为空，说明检测到了bbox，检测到了目标
                if len(det):
                    # bbox 放大到原始图像的大小，从img_size 到原始图像 im0 尺寸， bbox左上角的x1，y1, 右下角的x2,y2
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # eg: [[832.0, 160.0, 1495.0, 610.0, 0.9033942818641663], [849.0, 1918.0, 1467.0, 2016.0, 0.8640206456184387], [204.0, 0.0, 798.0, 142.0, 0.2842876613140106]]
                    bboxes = det[:, :4].tolist()
                    confidences = det[:, 4].tolist()
                    # eg: ['figure', 'equation', 'figure']
                    labels = [self.label_list_cn[i] for i in map(int, det[:, -1].tolist())]
                    #图片的名称，bboex，置信度，标签，都加到结果
                    results.append([images[idx], bboxes, confidences, labels])
                    #最后一个维度的最后一位是预测的结果, unique是为了统计多个相同的结果
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n}个{self.label_list[int(c)]}{'s' * (n > 1)} bbox, "  # eg: 2个figures bbox

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_img or self.view_img:  # 给图片添加bbox，画到图片上
                            label = f'{self.label_list[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    print(f'{s}完成. 耗时({t2 - t1:.3f}s)')
                else:
                    #如果没有匹配到，那么为空
                    results.append([images[idx], [], [], []])
                    print(f'{s}完成. 没有发现目标,耗时({t2 - t1:.3f}s)')

                # 保存图片的识别结果到目录下
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    print(f"保存识别结果到 {save_path}{s}")
        print(f'Done. ({time.time() - t0:.3f}s)')
        return results

    def baidu_ocr(self, image_byte):
        """
        返回json格式的预测结果
        :param image_byte, 图片的bytes格式
        :return: string 格式的识别结果
        """
        import base64
        sys.path.append('/opt/salt-daily-check/bin')
        from baidutoken import gettoken
        access_token = gettoken()
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
        # 二进制方式打开图片文件
        # 图片识别成文字
        results = ''
        img = base64.b64encode(image_byte)
        params = {"image": img}
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        if response.status_code == 200:
            res = response.json()
            if res.get("words_result") is None:
                print(f"错误: 百度OCR返回的消息是: {res}")
                sys.exit(0)
            for w in res['words_result']:
                results = results + w['words'] + '\n'
        return results

    def extract(self, detect_data, extract_dir):
        """
        对detect得到的结果，截取其中目标检测的内容，保存到extract_dir
        对于截取的图片的命名，需要通过OCR识别,需要调用baidu OCR的api
        :param detect_data: 是dectect函数的结果
        :param extract_dir: 提取图片中的表格，公式，图片，裁剪出来，保存到这个目录中
        :return:
        """
        for img_idx, data in enumerate(detect_data):
            images, bboxes, confidences, labels = data
            img = cv2.imread(images)
            for box_idx, (bbox, label) in enumerate(zip(bboxes, labels)):
                #每个候选框识别图片的结果
                x1, y1, x2, y2 = list(map(int, bbox))
                #截取图片
                crop_img = img[y1:y2, x1:x2]
                #识别图片，获取名字
                retval, buffer = cv2.imencode('.jpg', crop_img)
                img_bytes = buffer.tostring()
                ocr_res = self.baidu_ocr(image_byte=img_bytes)
                #图片名字
                en_label = self.label_list[self.label_list_cn.index(label)]
                ocr_res = ocr_res.lower()
                p = re.compile(r'(?<=\b%s )\d+\b' % en_label)
                if en_label in ['table', 'figure']:
                    res = re.findall(p, ocr_res)
                    if res:
                        name = f"image{img_idx}_{box_idx}_{en_label}_{res[0]}.jpg"
                    else:
                        name = f"image{img_idx}_{box_idx}_{en_label}_x1.jpg"
                else:
                    #是公式，那么命名的顺序是有区别的，例如识别的是 了≡Wqj-0+1:    (1)
                    num_res = re.findall('(?<=\()\d+(?=\))', ocr_res)
                    if num_res:
                        equation_num = num_res[-1]
                        name = f"image{img_idx}_{box_idx}_equation_{equation_num}.jpg"
                    else:
                        name = f"image{img_idx}_{box_idx}_equation_x2.jpg"
                #保存图片
                name_path = os.path.join(extract_dir,name)
                cv2.imwrite(name_path, crop_img)

    def do_train(self, data):
        """
        训练模型, 数据集分成2部分，训练集和验证集, 默认比例9:1
        :param data: 输入的数据，注意如果做truncated，那么输入的数据为 []
        :return:
        """
        pass
        logger.info(f"训练完成")
        return "Done"

@app.route("/api/predict", methods=['POST'])
def predict():
    """
    接收POST请求，获取data参数
    Args:
        test_data: 需要预测的数据，是一个图片的url列表, [images1, images2]
    Returns: 返回格式是[[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...]
    results = {list: 4} [['/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg', [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]], [0.9033942818641663, 0.8640206456184387, 0.2842876613140106], ['figure', 'equation', 'figure']], ['/Users/admin/git/yolov5/runs/api/images/A_Comprehensive_Survey_of_Grammar_Error_Correction0001-21.jpg', [[864.0, 132.0, 1608.0, 459.0], [865.0, 1862.0, 1602.0, 1944.0], [863.0, 1655.0, 1579.0, 1753.0], [115.0, 244.0, 841.0, 327.0], [116.0, 398.0, 837.0, 486.0], [124.0, 130.0, 847.0, 235.0], [119.0, 1524.0, 830.0, 1616.0], [161.0, 244.0, 799.0, 447.0]], [0.9183754920959473, 0.8920623660087585, 0.8884797692298889, 0.8873556852340698, 0.8276346325874329, 0.5401338934898376, 0.33260053396224976, 0.2832690477371216], ['table', 'equation', 'equation', 'equation', 'equation', 'equation', 'equation', 'equation']], ['/Users/admin/git/yolov5/runs/api/images/2007.158710001-09.jpg', [], ...
     0 = {list: 4} ['/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg', [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]], [0.9033942818641663, 0.8640206456184387, 0.2842876613140106], ['figure', 'equation', 'figure']]
      0 = {str} '/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg'
      1 = {list: 3} [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]]
      2 = {list: 3} [0.9033942818641663, 0.8640206456184387, 0.2842876613140106]
      3 = {list: 3} ['figure', 'equation', 'figure']
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.detect(test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/extract", methods=['POST'])
def extract():
    """
    接收POST请求，获取data参数
    Args:
        test_data: 需要预测的数据，是一个图片的url列表, [images1, images2]
    Returns: 返回格式是[[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...]
    results = {list: 4} [['/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg', [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]], [0.9033942818641663, 0.8640206456184387, 0.2842876613140106], ['figure', 'equation', 'figure']], ['/Users/admin/git/yolov5/runs/api/images/A_Comprehensive_Survey_of_Grammar_Error_Correction0001-21.jpg', [[864.0, 132.0, 1608.0, 459.0], [865.0, 1862.0, 1602.0, 1944.0], [863.0, 1655.0, 1579.0, 1753.0], [115.0, 244.0, 841.0, 327.0], [116.0, 398.0, 837.0, 486.0], [124.0, 130.0, 847.0, 235.0], [119.0, 1524.0, 830.0, 1616.0], [161.0, 244.0, 799.0, 447.0]], [0.9183754920959473, 0.8920623660087585, 0.8884797692298889, 0.8873556852340698, 0.8276346325874329, 0.5401338934898376, 0.33260053396224976, 0.2832690477371216], ['table', 'equation', 'equation', 'equation', 'equation', 'equation', 'equation', 'equation']], ['/Users/admin/git/yolov5/runs/api/images/2007.158710001-09.jpg', [], ...
     0 = {list: 4} ['/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg', [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]], [0.9033942818641663, 0.8640206456184387, 0.2842876613140106], ['figure', 'equation', 'figure']]
      0 = {str} '/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg'
      1 = {list: 3} [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]]
      2 = {list: 3} [0.9033942818641663, 0.8640206456184387, 0.2842876613140106]
      3 = {list: 3} ['figure', 'equation', 'figure']
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    extract_dir = jsonres.get('extract_dir', None)
    detect_data = model.detect(test_data)
    results = model.extract(detect_data=detect_data, extract_dir=extract_dir)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/train", methods=['POST'])
def train():
    """
    接收data参数，
    Args:
        data: 训练的数据，是一个图片列表, [images1, images2,...]
    Returns:
    """
    jsonres = request.get_json()
    data = jsonres.get('data', None)
    logger.info(f"收到的数据是:{data}, 进行训练")
    results = model.do_train(data)
    return jsonify(results)

if __name__ == "__main__":
    model = YOLOModel()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
