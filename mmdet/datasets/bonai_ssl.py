import csv
import math
import os.path as osp

import numpy as np

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class BONAI_SSL(CocoDataset):
    CLASSES = ("building",)

    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        edge_prefix=None,
        side_face_prefix=None,
        offset_field_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        gt_footprint_csv_file=None,
        bbox_type="roof",
        mask_type="roof",
        offset_coordinate="rectangle",
        resolution=0.6,
        ignore_buildings=True,
        height_mask_shape=[256, 256],
        image_scale_footprint_mask_shape=[256, 256],
    ):
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
        )
        self.ann_file = ann_file
        self.bbox_type = bbox_type
        self.mask_type = mask_type
        self.offset_coordinate = offset_coordinate
        self.resolution = resolution
        self.height_mask_shape = height_mask_shape
        self.image_scale_footprint_mask_shape = image_scale_footprint_mask_shape
        self.ignore_buildings = ignore_buildings
        self.gt_footprint_csv_file = gt_footprint_csv_file

        self.edge_prefix = edge_prefix
        self.side_face_prefix = side_face_prefix
        self.offset_field_prefix = offset_field_prefix

        if self.data_root is not None:
            if not (self.edge_prefix is None or osp.isabs(self.edge_prefix)):
                self.edge_prefix = osp.join(self.data_root, self.edge_prefix)
            if not (self.side_face_prefix is None or osp.isabs(self.side_face_prefix)):
                self.side_face_prefix = osp.join(self.data_root, self.side_face_prefix)
            if not (self.offset_field_prefix is None or osp.isabs(self.offset_field_prefix)):
                self.offset_field_prefix = osp.join(self.data_root, self.offset_field_prefix)

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results["edge_prefix"] = self.edge_prefix
        results["edge_fields"] = []

        results["side_face_prefix"] = self.side_face_prefix
        results["side_face_fields"] = []

        results["offset_field_prefix"] = self.offset_field_prefix
        results["offset_field_fields"] = []

    def get_properties(self, idx):
        """Get ann dict keys."""
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        return ann_info[0].keys()

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_["image_id"] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info["id"]
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all(_["iscrowd"] for _ in ann_info)
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann or all_iscrowd):
                continue
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks = []
        gt_roof_masks = []
        gt_footprint_masks = []
        gt_offsets = []
        gt_heights = []
        gt_nadir_angles = []
        gt_mean_nadir_angle = 0.0
        gt_offset_angles = []
        gt_mean_offset_angle = 0.0
        gt_roof_bboxes = []
        gt_footprint_bboxes = []
        gt_only_footprint_flag = 0

        # is_semi_supervised_sample = 1 if (self.bbox_type + "_bbox") not in ann_info[0] else 0
        is_semi_supervised_sample = 1 if "offset" not in ann_info[0] else 0
        is_valid_height_sample = 1 if "building_height" in ann_info[0] else 0
        # Variable fake_wh is used only for passing the valid check.
        fake_wh = 500

        for ann in ann_info:
            if ann.get("ignore", False):
                continue

            # TODO: remove the patch for wrong field name.
            if not "footprint_bbox" in ann:
                ann["footprint_bbox"] = ann["bbox"]
            if not "footprint_mask" in ann:
                ann["footprint_mask"] = ann["segmentation"][0]

            ann_category_id = ann["category_id"]
            if ann_category_id not in self.cat_ids:
                continue

            # bbox type may be roof, building and footprint, set it in the config file
            if self.bbox_type == "roof":
                x1, y1, w, h = ann.get(  # pylint: disable=invalid-name
                    "roof_bbox", [0, 0, fake_wh, fake_wh]
                )
            elif self.bbox_type == "building":
                x1, y1, w, h = ann.get(  # pylint: disable=invalid-name
                    "building_bbox", [0, 0, fake_wh, fake_wh]
                )
            elif self.bbox_type == "footprint":
                x1, y1, w, h = ann.get(  # pylint: disable=invalid-name
                    "footprint_bbox", [0, 0, fake_wh, fake_wh]
                )
            else:
                raise TypeError(f"unsupported bbox_type={self.bbox_type}")

            if ("area" in ann and ann["area"] <= 0) or w < 1 or h < 1:
                continue

            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False) and self.ignore_buildings:
                gt_bboxes_ignore.append(bbox)
                continue

            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann["category_id"]])
            gt_roof_masks.append(ann.get("segmentation", [[0, 0, 0, 0, 0, 0]]))
            gt_footprint_masks.append([ann["footprint_mask"]])

            if "roof_bbox" in ann:
                x1, y1, w, h = ann["roof_bbox"]  # pylint: disable=invalid-name
                gt_roof_bboxes.append([x1, y1, x1 + w, y1 + h])

            if "footprint_bbox" in ann:
                x1, y1, w, h = ann["footprint_bbox"]  # pylint: disable=invalid-name
                gt_footprint_bboxes.append([x1, y1, x1 + w, y1 + h])

            if self.mask_type == "roof":
                gt_masks.append(
                    ann.get(
                        "segmentation", [[0, 0, fake_wh, 0, fake_wh, fake_wh, 0, fake_wh, 0, 0]]
                    )
                )
            elif self.mask_type == "footprint":
                gt_masks.append([ann["footprint_mask"]])
            else:
                raise TypeError(f"unsupported mask_type={self.mask_type}")

            # rectangle coordinate -> offset = (x, y)
            # polar coordinate -> offset = (length, theta)
            if "offset" in ann:
                if self.offset_coordinate == "rectangle":
                    gt_offsets.append(ann["offset"])
                elif self.offset_coordinate == "polar":
                    offset_x, offset_y = ann["offset"]
                    length = math.sqrt(offset_x**2 + offset_y**2)
                    angle = math.atan2(offset_y, offset_x)
                    gt_offsets.append([length, angle])
                else:
                    raise RuntimeError(f"unsupported coordinate={self.offset_coordinate}")
            else:
                gt_offsets.append([5, 5])

            gt_height = ann.get("building_height", 0.0)
            gt_heights.append([gt_height])

            if "offset" in ann and "building_height" in ann:
                offset_x, offset_y = ann["offset"]
                norm = offset_x**2 + offset_y**2
                height = ann["building_height"]
                if height != 0 and norm != 0:
                    angle = math.sqrt(norm) * self.resolution / float(height)
                    gt_nadir_angles.append(angle)

            if "offset" in ann:
                offset = ann["offset"]
                if offset != [0, 0]:
                    z = math.sqrt(offset[0] ** 2 + offset[1] ** 2)
                    gt_offset_angles.append([float(offset[1]) / z, float(offset[0]) / z])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_roof_bboxes = np.array(gt_roof_bboxes, dtype=np.float32)
            gt_footprint_bboxes = np.array(gt_footprint_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_offsets = np.array(gt_offsets, dtype=np.float32)
            gt_heights = np.array(gt_heights, dtype=np.float32)
            gt_only_footprint_flag = float(gt_only_footprint_flag)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_roof_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_footprint_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_offsets = np.zeros((0, 2), dtype=np.float32)
            gt_heights = np.zeros((0, 1), dtype=np.float32)
            gt_only_footprint_flag = 0

        if len(gt_nadir_angles) > 0:
            gt_mean_nadir_angle = float(np.array(gt_nadir_angles, dtype=np.float32).mean())
        else:
            gt_mean_nadir_angle = 1

        if len(gt_offset_angles) > 0:
            gt_mean_offset_angle = [list(np.array(gt_offset_angles, dtype=np.float32).mean(axis=0))]
        else:
            gt_mean_offset_angle = [[1.0, 0.0]]

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info["filename"].replace("jpg", "png")
        edge_map = img_info["filename"].replace("jpg", "png")
        side_face_map = img_info["filename"].replace("jpg", "png")
        offset_field = img_info["filename"].replace("png", "npy")

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks,
            roof_masks=gt_roof_masks,
            footprint_masks=gt_footprint_masks,
            seg_map=seg_map,
            offsets=gt_offsets,
            building_heights=gt_heights,
            nadir_angles=gt_mean_nadir_angle,
            offset_angles=gt_mean_offset_angle,
            edge_map=edge_map,
            side_face_map=side_face_map,
            roof_bboxes=gt_roof_bboxes,
            footprint_bboxes=gt_footprint_bboxes,
            offset_field=offset_field,
            only_footprint_flag=gt_only_footprint_flag,
            is_semi_supervised_sample=is_semi_supervised_sample,
            is_valid_height_sample=is_valid_height_sample,
            resolution=self.resolution,
            height_mask_shape=np.array(self.height_mask_shape, dtype=np.int32),
            image_scale_footprint_mask_shape=np.array(
                self.image_scale_footprint_mask_shape, dtype=np.int32
            ),
        )

        return ann

    def _segm2json(self, results, mask_type=None):
        assert mask_type is not None
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            det = result[0]
            mask_type_idx = {
                "roof": 1,
                "offset_footprint": 4,
                "direct_footprint": 5,
            }
            try:
                seg = result[mask_type_idx[mask_type]]
            except KeyError as e:
                raise KeyError("wrong mask type to evaluate") from e

            if seg is None:
                break

            for i, bboxes in enumerate(det):
                # bbox results
                for j in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[j])
                    data["score"] = float(bboxes[j][4])
                    data["category_id"] = self.cat_ids[i]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][i]
                    mask_score = seg[1][i]
                else:
                    segms = seg[i]
                    mask_score = [bbox[4] for bbox in bboxes]
                for j in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[j])
                    data["score"] = float(mask_score[j])
                    data["category_id"] = self.cat_ids[i]
                    if isinstance(segms[j]["counts"], bytes):
                        segms[j]["counts"] = segms[j]["counts"].decode()
                    data["segmentation"] = segms[j]
                    segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def write_results2csv(self, results, meta_info=None):
        """Write results to csv file."""
        print("meta_info: ", meta_info)
        segmentation_eval_results = results[0]
        with open(meta_info["summary_file"], "w", encoding="utf-8") as summary:
            csv_writer = csv.writer(summary, delimiter=",")
            csv_writer.writerow(["Meta Info"])
            csv_writer.writerow(["model", meta_info["model"]])
            csv_writer.writerow(["anno_file", meta_info["anno_file"]])
            csv_writer.writerow(["gt_roof_csv_file", meta_info["gt_roof_csv_file"]])
            csv_writer.writerow(["gt_footprint_csv_file", meta_info["gt_footprint_csv_file"]])
            csv_writer.writerow(["vis_dir", meta_info["vis_dir"]])
            csv_writer.writerow([""])
            for mask_type in ["roof", "footprint"]:
                csv_writer.writerow([mask_type])
                csv_writer.writerow([segmentation_eval_results[mask_type]])
                csv_writer.writerow(["F1 Score", segmentation_eval_results[mask_type]["F1_score"]])
                csv_writer.writerow(
                    ["Precision", segmentation_eval_results[mask_type]["Precision"]]
                )
                csv_writer.writerow(["Recall", segmentation_eval_results[mask_type]["Recall"]])
                csv_writer.writerow(["True Positive", segmentation_eval_results[mask_type]["TP"]])
                csv_writer.writerow(["False Positive", segmentation_eval_results[mask_type]["FP"]])
                csv_writer.writerow(["False Negative", segmentation_eval_results[mask_type]["FN"]])
                csv_writer.writerow([""])

            csv_writer.writerow([""])

    def _set_group_flag(self):
        """Set flag according to image aspect ratio and whether the image is fully annotated.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info["width"] / img_info["height"] > 1:
                if "offset" in self.get_properties(i):
                    self.flag[i] = 0
                else:
                    if "building_height" in self.get_properties(i):
                        self.flag[i] = 1
                    else:
                        self.flag[i] = 2
            else:
                if "offset" in self.get_properties(i):
                    self.flag[i] = 3
                else:
                    if "building_height" in self.get_properties(i):
                        self.flag[i] = 4
                    else:
                        self.flag[i] = 5
