# -*- encoding: utf-8 -*-
'''
@File    :   vis_offset_filed.py
@Time    :   2020/12/30 22:35:15
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 offset field
'''

import numpy as np
import cv2
import os
import bstool
import mmcv


if __name__ == '__main__':

    offset_field_dir = './data/buildchange/v0/shanghai/arg/offset_field/Npy'

    for fn in os.listdir(offset_field_dir):
        basename = bstool.get_basename(fn)

        shp_file = f'./data/buildchange/v0/shanghai/arg/roof_shp_4326/{basename}.shp'
        geo_file = f'./data/buildchange/v0/shanghai/arg/geo_info/{basename}.png'
        rgb_file = f'./data/buildchange/v0/shanghai/arg/images/{basename}.jpg'

        if not os.path.exists(shp_file):
            continue
        
        rgb = cv2.imread(rgb_file)

        offset_field = np.load(os.path.join(offset_field_dir, fn))

        offset_x, offset_y = offset_field[..., 0], offset_field[..., 1]

        offset_x = mmcv.imrescale(
                    offset_x,
                    1.0,
                    interpolation='nearest',
                    backend='cv2')

        offset_x, offset_y = offset_x.astype(np.int), offset_y.astype(np.int)

        XX, YY = np.meshgrid(np.arange(0, rgb.shape[1]), np.arange(0, rgb.shape[0]))

        x_moved_coordinate = offset_x + XX
        y_moved_coordinate = offset_y + YY

        x_moved_coordinate = np.clip(x_moved_coordinate, 0, rgb.shape[1] - 1)
        y_moved_coordinate = np.clip(y_moved_coordinate, 0, rgb.shape[0] - 1)

        objects = bstool.shp_parse(shp_file=shp_file,
                                    geo_file=geo_file,
                                    src_coord='4326',
                                    dst_coord='pixel')

        polygons = [obj['polygon'] for obj in objects]

        roof_mask = bstool.generate_image(2048, 2048, (0, 0, 0))
        for idx, polygon in enumerate(polygons):
            mask = bstool.polygon2mask(polygon)
            mask = np.array(mask).reshape(1, -1, 2)
            cv2.fillPoly(roof_mask, mask, (255, 255, 255))

        footprint_mask = roof_mask[y_moved_coordinate, x_moved_coordinate]

        fusion = cv2.addWeighted(footprint_mask, 0.4, rgb, 0.6, 0.0)

        bstool.show_image(fusion)