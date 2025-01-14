# -*- encoding: utf-8 -*-
'''
@File    :   convert2wireframe.py
@Time    :   2020/12/19 23:01:30
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   将过渡格式转换成 wireframe 格式，用于训练线段检测模型或者线框解析模型
'''


import os
import numpy as np
import mmcv
import json

import bstool


if __name__ == '__main__':
    image_dir = './data/buildchange/v1/jinan/images'
    label_dir = './data/buildchange/v1/jinan/labels'
    json_file = './data/buildchange/v1/jinan/wireframe/train.json'

    bstool.mkdir_or_exist('./data/buildchange/v1/jinan/wireframe/')

    height, width = 1024, 1024

    json_data = []
    progress_bar = mmcv.ProgressBar(len(os.listdir(label_dir)))
    for label_fn in os.listdir(label_dir):
        image_anns = {}
        label_file = os.path.join(label_dir, label_fn)

        ori_anns = mmcv.load(label_file)['annotations']

        image_anns['width'] = width
        image_anns['height'] = height

        junctions, edges_positive, edges_negative = [], [], []
        masks = []
        for ori_ann in ori_anns:
            mask = ori_ann['roof']
            masks.append(mask)
        
        if len(masks) < 5:
            continue
        
        wireframe_items = bstool.mask2wireframe(masks)
        junctions += wireframe_items[0]
        edges_positive += wireframe_items[1]
        edges_negative += wireframe_items[2]

        # bstool.show_masks_on_image(bstool.generate_image(1024, 1024), masks)
        
        image_anns['junctions'] = junctions
        image_anns['edges_positive'] = edges_positive
        image_anns['edges_negative'] = edges_negative
        image_anns['filename'] = bstool.get_basename(label_fn) + '.png'

        json_data.append(image_anns)

        progress_bar.update()

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent = 4)