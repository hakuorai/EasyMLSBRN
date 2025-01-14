import bstool
import mmcv
import json


if __name__ == '__main__':
    for imageset in ['train', 'val']:
        src_json_file = f'./data/urban3d/v2/coco/annotations/urban3d_v2_{imageset}_JAX_OMA.json'
        # src_json_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_train_samples.json'
        dst_json_file = f'./data/urban3d/v2/coco/annotations/urban3d_v2_{imageset}_JAX_OMA_reverse_offset_roof2footprint.json'
        # dst_json_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_train_samples_compression.json'

        src_data = mmcv.load(src_json_file)
        src_annotations = src_data['annotations']

        dst_annotations = []
        for src_annotation in src_annotations:
            offset = src_annotation['offset']
            src_annotation['offset'] = [-offset[0], -offset[1]]
            dst_annotations.append(src_annotation)

        src_data['annotations'] = dst_annotations

        with open(dst_json_file, "w") as jsonfile:
            json.dump(src_data, jsonfile, sort_keys=True, indent=4)