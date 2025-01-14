import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':

    image_dir = '/data/urban3d/v1/val/images'
    # csv_df = pandas.read_csv('/data/urban3d/weijia/urban3d_jax_oma_val_orgfootprint_offset_gt_simple_subcsv_merge.csv')
    csv_df = pandas.read_csv('/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v008.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_urban3d/bc_v008.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_urban3d_footprint_merged.csv')


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