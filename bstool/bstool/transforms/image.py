import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
import geopandas
import matplotlib.pyplot as plt
import mmcv

import bstool


def split_image(img, 
                subsize=1024, 
                gap=200, 
                mode='keep_all', 
                expand_boundary=True):
    """split the original image to sub-images

    Args:
        img (np.array): input image
        subsize (float, optional): the size of sub-image. Defaults to 1024.
        gap (int, optional): the gap of sliding windows. Defaults to 200.
        mode (str, optional): keep all or drop boundary. Defaults to 'keep_all'.
        expand_boundary (bool, optional): if original image size < subsize, boundary will be expanded. Defaults to True.

    Returns:
        dict: splitted sub-images
    """
    if isinstance(img, str):
        img = cv2.imread(img)

    if img is None:
        print("This image is empty")
        return None
    
    img_height, img_width = img.shape[0], img.shape[1]

    start_xs = np.arange(0, img_width, subsize - gap)
    if mode == 'keep_all':
        start_xs[-1] = img_width - subsize if img_width - start_xs[-1] <= subsize else start_xs[-1]
    elif mode == 'drop_boundary':
        if img_width - start_xs[-1] < subsize - gap:
            start_xs = np.delete(start_xs, -1)
    start_xs[-1] = np.maximum(start_xs[-1], 0)

    start_ys = np.arange(0, img_height, subsize - gap)
    if mode == 'keep_all':
        start_ys[-1] = img_height - subsize if img_height - start_ys[-1] <= subsize else start_ys[-1]
    elif mode == 'drop_boundary':
        if img_height - start_ys[-1] < subsize - gap:
            start_ys = np.delete(start_ys, -1)
    start_ys[-1] = np.maximum(start_ys[-1], 0)

    subimages = dict()
    
    for start_x in start_xs:
        for start_y in start_ys:
            end_x = np.minimum(start_x + subsize, img_width)
            end_y = np.minimum(start_y + subsize, img_height)
            if expand_boundary:
                subimage = bstool.generate_image(subsize, subsize, color=(0, 0, 0))
                subimage[0:end_y-start_y, 0:end_x-start_x, ...] = img[start_y:end_y, start_x:end_x, ...]
            else:
                subimage = img[start_y:end_y, start_x:end_x, ...]
            coordinate = (start_x, start_y)
            subimages[coordinate] = subimage

    return subimages


def drop_subimage(subimages, 
                  subimage_coordinate, 
                  subimage_polygons,
                  center_area=2, 
                  small_object=96,
                  image_size=(1024, 1024),
                  show=False):
    """judge whether to drop the overlap image

    Arguments:
        subimages {dict} -- dict which contains all subimages (value)
        subimage_coordinate {tuple} -- the coordinate of subimage in original image
        subimage_masks {list} -- list of masks in subimages

    Keyword Arguments:
        center_area {int} -- the area of center line (default: {2})
        show {bool} -- whether to show center line (default: {False})

    Returns:
        drop flag -- True: drop the subimage, False: keep the subimage
    """
    # black image
    if np.mean(subimages[subimage_coordinate]) == 0:
        return True

    subimage_polygons = bstool.clean_polygon(subimage_polygons)

    # no object
    if len(subimage_polygons) == 0:
        return True

    # keep the main subimage, just drop the overlap part
    if abs(subimage_coordinate[0] - subimage_coordinate[1]) in (0, 1024) and (subimage_coordinate[0] != 512 and subimage_coordinate[1] != 512):
        return False

    # (horizontal, vertical)
    center_lines = [Polygon([(0, 512 - center_area), 
                            (0, 512 + center_area), 
                            (1023, 512 + center_area), 
                            (1023, 512 - center_area), 
                            (0, 512 - center_area)]), 
                    Polygon([(512 - center_area, 0), 
                            (512 + center_area, 0), 
                            (512 + center_area, 1023), 
                            (512 - center_area, 1023), 
                            (512 - center_area, 0)])]

    if subimage_coordinate[0] == 512 and subimage_coordinate[1] != 512:
        center_lines = [center_lines[1]]
    elif subimage_coordinate[0] != 512 and subimage_coordinate[1] == 512:
        center_lines = [center_lines[0]]
    else:
        center_lines = center_lines

    subimage_polygon_df = geopandas.GeoDataFrame({'geometry': subimage_polygons, 'submask_df':range(len(subimage_polygons))})
    center_line_df = geopandas.GeoDataFrame({'geometry': center_lines, 'center_df':range(len(center_lines))})

    image_boundary = [Polygon([(0, 0), (1024 - 1, 0), (1024 - 1, 1024 - 1), (0, 1024 - 1), (0, 0)])]
    border_line_df = geopandas.GeoDataFrame({'geometry': image_boundary, 'border_df':range(len(image_boundary))})

    if show:
        fig, ax = plt.subplots()   

        subimage_polygon_df.plot(ax=ax, color='red')
        center_line_df.plot(ax=ax, facecolor='none', edgecolor='g')
        border_line_df.plot(ax=ax, facecolor='none', edgecolor='k')
        ax.set_title('{}_{}'.format(subimage_coordinate[0], subimage_coordinate[1]))
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.gca().invert_yaxis()

        plt.show()

    subimage_polygon_df = subimage_polygon_df.loc[~subimage_polygon_df.geometry.is_empty]
    center_line_df = center_line_df.loc[~center_line_df.geometry.is_empty]
    
    res_intersection = geopandas.overlay(subimage_polygon_df, center_line_df, how='intersection')
    inter_dict = res_intersection.to_dict()
    ignore_indexes = list(set(inter_dict['submask_df'].values()))

    inter_areas = []
    for ignore_index in ignore_indexes:
        inter_areas.append(subimage_polygons[ignore_index].area)

    if len(inter_areas) == 0:
        return True
    elif len(inter_areas) < 5:
        if max(inter_areas) < small_object * small_object:
            return True
        else:
            return False
    else:
        if max(inter_areas) < small_object * small_object / 4:
            return True
        else:
            return False

def image_flip(img, transform_flag='h'):
    """flip the image

    Args:
        img (np.array): input image
        transform_flag (str, optional): horizontal or vertical. Defaults to 'h'.

    Raises:
        NotImplementedError: [description]

    Returns:
        np.array: flipped image
    """
    if transform_flag == 'h':
        img = cv2.flip(img, 1)
    elif transform_flag == 'v':
        img = cv2.flip(img, 0)
    else:
        raise NotImplementedError

    return img

def image_rotate(img, angle=0):
    """rotate image

    Args:
        img (np.array): 
        angle (int, optional): rotate angle. Defaults to 0. (rad/s)

    Returns:
        np.array: rotated image
    """
    angle = angle * 180.0 / np.pi
    img = mmcv.imrotate(img, angle)

    return img

def image_translation(img, offset_x, offset_y, border_value=0):
    """translate image

    Args:
        img (np.array): input image
        offset_x (int or float): translation distance in the x
        offset_y (int or float): translation distance in the y
        border_value (int, optional): [description]. Defaults to 0.
    """
    h, w = img.shape[:2]

    matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

    translated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    
    return translated