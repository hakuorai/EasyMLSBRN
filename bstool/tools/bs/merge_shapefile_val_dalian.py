# -*- encoding: utf-8 -*-
'''
@File    :   merge_shapefile_val_dalian.py
@Time    :   2020/12/30 22:40:18
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   合并分块标注的 shapefile 文件
'''


import os
import glob
import tqdm
import pandas
import geopandas
from multiprocessing import Pool
from functools import partial

import bstool

pandas.set_option('display.max_rows', None)

class MergeShapefile():
    def __init__(self,
                core_dataset_name,
                src_version,
                city,
                multi_processing=False,
                num_processor=8):
        self.image_path = f'./data/{core_dataset_name}/{src_version}/{city}/images'
        self.anno_path = f'./data/{core_dataset_name}/{src_version}/{city}/footprint_shp_pixel'
        self.geo_path = f'./data/{core_dataset_name}/{src_version}/{city}/images'

        self.merged_shapefile_save_path = f'./data/{core_dataset_name}/{src_version}/{city}/merged_shp'
        bstool.mkdir_or_exist(self.merged_shapefile_save_path)

        self.core_dataset_name = core_dataset_name
        self.src_version = src_version
        self.city = city
        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def merge_shapefile(self, shp_file):
        file_name = bstool.get_basename(shp_file)
        image_file = os.path.join(self.image_path, file_name + '.jpg')

        merged_shapefile = os.path.join(self.merged_shapefile_save_path, file_name + '.shp')

        if os.path.exists(merged_shapefile):
            return
            
        geo_file = os.path.join(self.geo_path, file_name + '.jpg')

        objects = bstool.shp_parse(shp_file, 
                                    geo_file,
                                    src_coord='pixel',
                                    dst_coord='pixel')
        if objects == []:
            return

        polygons = [obj['polygon'] for obj in objects]
        properties = [obj['property'] for obj in objects]

        merged_polygons, merged_properties = bstool.merge_polygons(polygons, properties)

        properties = []
        for idx, merged_property in enumerate(merged_properties):
            merged_property['Id'] = idx
            properties.append(merged_property)
            
        df = pandas.DataFrame(properties)
        gdf = geopandas.GeoDataFrame(df, geometry=merged_polygons, crs=None)
        gdf.to_file(merged_shapefile, encoding='utf-8')

    def core(self):
        shp_file_list = glob.glob("{}/*.shp".format(self.anno_path))
        num_image = len(shp_file_list)
        if self.multi_processing:
            worker = partial(self.merge_shapefile)
            ret = list(tqdm.tqdm(self.pool.imap(worker, shp_file_list), total=num_image))
        else:
            shp_file_list = glob.glob("{}/*.shp".format(self.anno_path))
            for shp_file in tqdm.tqdm(shp_file_list):
                self.merge_shapefile(shp_file)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 

if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    cities = ['dalian_fine']

    for city in cities:
        print("Begin processing {} set.".format(city))
        merge_shapefile = MergeShapefile(core_dataset_name=core_dataset_name,
                                        src_version=src_version,
                                        city=city,
                                        multi_processing=True,
                                        num_processor=8)
        merge_shapefile.core()
        print("Finish processing {} set.".format(city))
            
