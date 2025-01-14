import os
import bstool


if __name__ == "__main__":
    imagesets = ['val', 'train']
    folds = ['jax', 'oma']

    for imageset in imagesets:
        json_dir = f'./data/urban3d/v1/{imageset}/weijia_labels'
        bstool.mkdir_or_exist(json_dir)
        for fold in folds:
            print(f"========== processing {imageset} {fold} ==========")
            csv_file = f'./data/urban3d/weijia/instance_gt_{imageset}/urban3d_{fold}_{imageset}_roof_offset_gt_simple_subcsv_merge.csv'
            csv_parser = bstool.CSVParse(csv_file, check_valid=False)
            image_name_list = csv_parser.image_name_list

            for image_name in image_name_list:
                # if image_name != 'JAX_Tile_163_RGB_001':
                #     continue
                # print(f'instance_gt_{imageset}/urban3d_{fold}_{imageset}_roof_offset_gt_subcsv_merge.csv')
                json_file = os.path.join(json_dir, f'{image_name}.json')
                image_info = {"ori_filename": f'{image_name}.tif',
                              "width": 2048,
                              "height": 2048}
                objects = csv_parser(image_name)
                if len(objects) == 0:
                    continue
                roof_polygons = []
                properties = []
                for obj in objects:
                    roof_polygon = obj['polygon']

                    data = dict()
                    data['offset'] = obj['offset']
                    data['building_height'] = obj['height']

                    roof_polygons.append(roof_polygon)
                    properties.append(data)

                # bstool.show_polygon(roof_polygons)

                bstool.urban3d_json_dump(roof_polygons, properties, image_info, json_file)


