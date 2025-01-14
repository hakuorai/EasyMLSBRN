# -*- encoding: utf-8 -*-
'''
@File    :   vis_csv.py
@Time    :   2020/12/30 22:05:45
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   对 CSV 文件进行可视化
'''

import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':

    image_dir = '/data/buildchange/v0/xian_fine/images'
    # csv_df = pandas.read_csv('/data/urban3d/weijia/urban3d_jax_oma_val_orgfootprint_offset_gt_simple_subcsv_merge.csv')
    csv_df = pandas.read_csv('/data/buildchange/v0/xian_fine/xian_val_footprint_gt_minarea100_26.csv')


    for image_name in os.listdir(image_dir):
        image_file = os.path.join(image_dir, image_name)
        image_basename = bstool.get_basename(image_name)

        img = cv2.imread(image_file)
 
        roof_masks = []
        for idx, row in csv_df[csv_df.ImageId == image_basename].iterrows():
            if type(row.PolygonWKT_Pix) == str:
                roof_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            else:
                roof_polygon = row.PolygonWKT_Pix

            roof_mask = bstool.polygon2mask(roof_polygon)
            roof_masks.append(roof_mask)

        if len(roof_masks) == 0:
            continue
        
        img = bstool.draw_masks_boundary(img, roof_masks)
        bstool.show_image(img)