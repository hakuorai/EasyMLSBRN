import os
import os.path as osp
import tempfile
import pandas
import csv

import mmcv
import bstool
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from terminaltables import AsciiTable
import tqdm


class SemanticEval():
    def __init__(self, 
                 results,
                 csv_gt_file='./data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv',
                 csv_pred_prefix=None,
                 score_threshold=0.3):
        self.csv_gt_file = csv_gt_file
        self.csv_pred_prefix = csv_pred_prefix
        self.score_threshold = score_threshold
        self.csv_pred_file = self.csv_pred_prefix + '.csv'
        if isinstance(results, str):
            self.results = mmcv.load(results)
        else:
            self.results = results
        
    def results2csv(self):
        first_in = True
        for image_name in tqdm.tqdm(list(self.results.keys())):
            bboxes, masks, scores = self.results[image_name]
            polygons = []
            for mask, score in zip(masks, scores):
                if score < self.score_threshold:
                    continue
                polygons.append(bstool.mask2polygon(mask))

            csv_image = pandas.DataFrame({'ImageId': image_name,
                                          'BuildingId': range(len(polygons)),
                                          'PolygonWKT_Pix': polygons,
                                          'Confidence': 1})
            if first_in:
                csv_dataset = csv_image
                first_in = False
            else:
                csv_dataset = csv_dataset.append(csv_image)

        csv_dataset.to_csv(self.csv_pred_file, index=False)

    def evaluation(self):
        print("Begin to convert results to csv files")
        self.results2csv()
        print("Begin to evaluate csv files")
        