# -*- encoding: utf-8 -*-
'''
@File    :   shapefile2csv_val_dalian.py
@Time    :   2020/12/30 22:40:50
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   将 shapefile 转换为 CSV 文件
'''


import os
import bstool
import numpy as np
import rasterio as rio
import cv2
import pandas
import glob
from shapely import affinity
import math


if __name__ == '__main__':
    roof_csv_file = './data/buildchange/v0/dalian_fine/dalian_fine_2048_roof_gt.csv'
    footprint_csv_file = './data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'
    
    first_in = True
    min_area = 100

    shp_dir = f'./data/buildchange/v0/dalian_fine/merged_shp'
    rgb_img_dir = f'./data/buildchange/v0/dalian_fine/images'

    shp_file_list = glob.glob("{}/*.shp".format(shp_dir))
    for shp_file in shp_file_list:
        base_name = bstool.get_basename(shp_file)

        rgb_img_file = os.path.join(rgb_img_dir, base_name + '.jpg')

        objects = bstool.shp_parse(shp_file=shp_file,
                                    geo_file=rgb_img_file,
                                    src_coord='pixel',
                                    dst_coord='pixel',
                                    keep_polarity=False)

        roof_gt_polygons, gt_properties, gt_heights, gt_offsets = [], [], [], []
        for obj in objects:
            roof_gt_polygon = obj['polygon']
            if roof_gt_polygon.area < min_area:
                continue
            valid_flag = bstool.single_valid_polygon(roof_gt_polygon)
            if not valid_flag:
                continue
            roof_gt_polygons.append(obj['polygon'])
            gt_offsets.append([obj['property']['xoffset'], obj['property']['yoffset']])
            if obj['property']['Floor'] is not None:
                if math.isnan(obj['property']['Floor']):
                    gt_heights.append(1 * 3)
                else:
                    gt_heights.append(obj['property']['Floor'] * 3)
            else:
                gt_heights.append(1 * 3)
            gt_properties.append(obj['property'])

        footprint_gt_polygons = bstool.roof2footprint(roof_gt_polygons, gt_properties)

        roof_csv_image = pandas.DataFrame({'ImageId': base_name,
                                        'BuildingId': range(len(roof_gt_polygons)),
                                        'PolygonWKT_Pix': roof_gt_polygons,
                                        'Confidence': 1,
                                        'Offset': gt_offsets,
                                        'Height': gt_heights})
        footprint_csv_image = pandas.DataFrame({'ImageId': base_name,
                                        'BuildingId': range(len(footprint_gt_polygons)),
                                        'PolygonWKT_Pix': footprint_gt_polygons,
                                        'Confidence': 1,
                                        'Offset': gt_offsets,
                                        'Height': gt_heights})
        if first_in:
            roof_csv_dataset = roof_csv_image
            footprint_csv_dataset = footprint_csv_image
            first_in = False
        else:
            roof_csv_dataset = roof_csv_dataset.append(roof_csv_image)
            footprint_csv_dataset = footprint_csv_dataset.append(footprint_csv_image)

    roof_csv_dataset.to_csv(roof_csv_file, index=False)
    footprint_csv_dataset.to_csv(footprint_csv_file, index=False)