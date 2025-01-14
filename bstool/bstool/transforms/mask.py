import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import geojson
import networkx as nx
import rasterio as rio
import geopandas
from shapely import affinity
import itertools

import bstool


def polygon2mask(polygon):
    """convet polygon to mask

    Arguments:
        polygon {Polygon} -- input polygon (single polygon)

    Returns:
        list -- converted mask ([x1, y1, x2, y2, ...])
    """
    mask = np.array(polygon.exterior.coords, dtype=int)[:-1].ravel().tolist()
    return mask

def mask2polygon(mask):
    """convert mask to polygon

    Arguments:
        mask {list} -- contains coordinates of mask boundary ([x1, y1, x2, y2, ...])
    """
    mask_x = mask[0::2]
    mask_y = mask[1::2]
    mask_coord = [(int(x), int(y)) for x, y in zip(mask_x, mask_y)]

    polygon = Polygon(mask_coord)

    return polygon

def mask2bbox(mask):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, ...]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(mask[0::2])
    ymin = min(mask[1::2])
    xmax = max(mask[0::2])
    ymax = max(mask[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def polygon_coordinate_convert(polygon, 
                               geo_img, 
                               src_coord='4326', 
                               dst_coord='pixel',
                               keep_polarity=True):
        """convert polygon of source coordinate to destination coordinate

        Args:
            polygon (Polygon): source polygon
            geo_img (str or rio class): coordinate infomation
            src_coord (str, optional): source coordinate. Defaults to '4326'.
            dst_coord (str, optional): destination coordinate. Defaults to 'pixel'.

        Returns:
            Polygon: converted polygon
        """
        if isinstance(geo_img, str):
            geo_img = rio.open(geo_img)

        if src_coord == '4326':
            if dst_coord == '4326':
                converted_polygon = polygon
            elif dst_coord == 'pixel':
                converted_polygon = [(geo_img.index(c[0], c[1])[1], geo_img.index(c[0], c[1])[0]) for c in polygon.exterior.coords]
                converted_polygon = Polygon(converted_polygon)
            else:
                raise(RuntimeError("Not support dst_coord = {}".format(dst_coord)))
        elif src_coord == 'pixel':
            if keep_polarity:
                converted_polygon = polygon
            else:
                converted_polygon = [(abs(geo_img.index(c[0], c[1])[1]), abs(geo_img.index(c[0], c[1])[0])) for c in polygon.exterior.coords]
                converted_polygon = Polygon(converted_polygon)
        else:
            raise(RuntimeError("Not support src_coord = {}".format(src_coord)))

        return converted_polygon

def merge_polygons(polygons, 
                   properties=None, 
                   connection_mode='floor',
                   merge_boundary=False):
        """merge polygons by floor

        Args:
            polygons (list): list of polygons
            properties (list, optional): list of properties. Defaults to None.
            connection_mode (str, optional): merge by which item. Defaults to 'floor'.
            merge_boundary (bool, optional): whether to merge boundary. Defaults to False.

        Returns:
            list: merged polygons
        """
        floors = []
        for single_property in properties:
            if 'Floor' in single_property.keys():
                floors.append(single_property['Floor'])
            elif 'half_H' in single_property.keys():
                floors.append(single_property['half_H'])
            else:
                return polygons, properties
                # print("Warning: No Floor key in property, keys = {}".format(single_property.keys()))

        merged_polygons = []
        merged_properties = []

        num_polygon = len(polygons)
        node_list = range(num_polygon)
        link_list = []
        for i in range(num_polygon):
            for j in range(i + 1, num_polygon):
                polygon1, polygon2 = polygons[i], polygons[j]
                floor1, floor2 = floors[i], floors[j]
                if polygon1.is_valid and polygon2.is_valid:
                    inter = polygon1.intersection(polygon2)

                    if connection_mode == 'floor':
                        if (inter.area > 0.0 or inter.geom_type in ["LineString", "MULTILINESTRING"]) and floor1 == floor2:
                            link_list.append([i, j])
                    elif connection_mode == 'line':
                        if inter.geom_type in ["LineString", "MULTILINESTRING"]:
                            link_list.append([i, j])
                    else:
                        raise(RuntimeError("Wrong connection_mode flag: {}".format(connection_mode)))

                    if merge_boundary:
                        if polygon1.area > polygon2.area:
                            difference = polygon1.difference(polygon2)
                            polygons[i] = difference
                        else:
                            difference = polygon2.difference(polygon1)
                            polygons[j] = difference
                else:
                    continue
        
        G = nx.Graph()
        for node in node_list:
            G.add_node(node, properties=properties[node])

        for link in link_list:
            G.add_edge(link[0], link[1])

        for c in nx.connected_components(G):
            nodeSet = G.subgraph(c).nodes()
            nodeSet = list(nodeSet)

            if len(nodeSet) == 1:
                single_polygon = polygons[nodeSet[0]]
                merged_polygons.append(single_polygon)
                merged_properties.append(properties[nodeSet[0]])
            else:
                pre_merge = [polygons[node] for node in nodeSet]
                post_merge = unary_union(pre_merge)
                if post_merge.geom_type == 'MultiPolygon':
                    for sub_polygon in post_merge:
                        merged_polygons.append(sub_polygon)
                        merged_properties.append(properties[nodeSet[0]])
                else:
                    merged_polygons.append(post_merge)
                    merged_properties.append(properties[nodeSet[0]])
        
        return merged_polygons, merged_properties


def get_ignored_polygon_idx(origin_polygons, ignore_polygons):
    """get index which should be ignored

    Args:
        origin_polygons (list): list of polygons
        ignore_polygons (list): list of polygons

    Returns:
        list: list of indexes
    """
    origin_polygons = geopandas.GeoSeries(origin_polygons)
    ignore_polygons = geopandas.GeoSeries(ignore_polygons)

    origin_df = geopandas.GeoDataFrame({'geometry': origin_polygons, 'foot_df':range(len(origin_polygons))})
    ignore_df = geopandas.GeoDataFrame({'geometry': ignore_polygons, 'ignore_df':range(len(ignore_polygons))})

    origin_df = origin_df.loc[~origin_df.geometry.is_empty]
    ignore_df = ignore_df.loc[~ignore_df.geometry.is_empty]

    res_intersection = geopandas.overlay(origin_df, ignore_df, how='intersection')
    inter_dict = res_intersection.to_dict()
    ignore_indexes = list(set(inter_dict['foot_df'].values()))
    ignore_indexes.sort()

    return ignore_indexes

def add_ignore_flag_in_property(properties, ignore_indexes):
    """add ignore flags to properties

    Args:
        properties (list): list of property
        ignore_indexes (list): list of index

    Returns:
        list: list of new properties
    """
    ret_properties = []
    for idx, single_property in enumerate(properties):
        if idx in ignore_indexes:
            single_property['ignore'] = 1
        else:
            single_property['ignore'] = 0
        ret_properties.append(single_property)
    return ret_properties

def get_polygon_centroid(polygons):
    """calculate the centroids of polygons

    Args:
        polygons (list): list of polygons

    Returns:
        list: list of centroids
    """
    centroids = []
    for polygon in polygons:
        coordinate = list(polygon.centroid.coords)[0]
        centroids.append(coordinate)

    return centroids

def select_polygons_in_range(polygons, coordinate, image_size=(1024, 1024)):
    """select polygons in specific range

    Args:
        polygons (list): list of polygon
        coordinate (list of tuple): given coordinate
        image_size (tuple, optional): image size. Defaults to (1024, 1024).

    Returns:
        np.array, bool: bool value for keeping which polygons
    """
    origin_centroids = get_polygon_centroid(polygons)
    origin_centroids = np.array(origin_centroids)
    converted_centroids = origin_centroids.copy()

    converted_centroids[:, 0] = origin_centroids[:, 0] - coordinate[0]
    converted_centroids[:, 1] = origin_centroids[:, 1] - coordinate[1]

    cx_bool = np.logical_and(converted_centroids[:, 0] >= 0, converted_centroids[:, 0] < image_size[0])
    cy_bool = np.logical_and(converted_centroids[:, 1] >= 0, converted_centroids[:, 1] < image_size[1])

    return np.logical_and(cx_bool, cy_bool)

def chang_polygon_coordinate(polygons, coordinate):
    """change the coordinate of polygon

    Args:
        polygons (list): list of polygon
        coordinate (list or tuple): distance of moving

    Returns:
        list: list of polygons
    """
    transform_matrix = [1, 0, 0, 1, -coordinate[0], -coordinate[1]]
    converted_polygons = []
    for polygon in polygons:
        transformed_polygon = affinity.affine_transform(polygon, transform_matrix)
        converted_polygons.append(transformed_polygon)
    
    return converted_polygons

def chang_mask_coordinate(masks, coordinate):
    """change the coordinate of mask

    Args:
        masks (np.array): (N, *) list of masks, np.array(list, list, list)
        coordinate (list or tuple): distance of moving

    Returns:
        list: list of masks
    """
    converted_masks = []
    for mask in masks:
        converted_mask = []
        for idx, value in enumerate(mask):
            if idx % 2 == 0:
                value = value + coordinate[0]
            else:
                value = value + coordinate[1]
            
            converted_mask.append(value)
        converted_masks.append(converted_mask)

    return converted_masks

def clip_boundary_polygon(polygons, image_size=(1024, 1024)):
    """clip the polygons to image boundary

    Args:
        polygons (list): list of polygons
        image_size (tuple, optional): the size of image. Defaults to (1024, 1024).

    Returns:
        list: list of polygons
    """
    h, w = image_size
    image_boundary = Polygon([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1), (0, 0)])

    polygons = geopandas.GeoDataFrame({'geometry': polygons, 'polygon_df':range(len(polygons))})

    clipped_polygons = geopandas.clip(polygons, image_boundary).to_dict()
    clipped_polygons = list(clipped_polygons['geometry'].values())
    return clipped_polygons

def clean_polygon(polygons):
    """convert polygon to valid polygon

    Arguments:
        polygons {list} -- list of polygon

    Returns:
        list -- cleaned polygons
    """
    polygons_ = []
    for polygon in polygons:
        if not polygon.is_valid:
            continue
        if type(polygon) == MultiPolygon:
            for single_polygon in polygon:
                if len(list(single_polygon.exterior.coords)) < 3:
                    continue
                if single_polygon.area < 5:
                    continue
                polygons_.append(single_polygon)
        elif type(polygon) == Polygon:
            if len(list(polygon.exterior.coords)) < 3:
                continue
            if polygon.area < 5:
                continue
            polygons_.append(polygon)
        else:
            continue

    return polygons_

def roof2footprint(roof_polygons, properties):
    """convert roof to polygon by properties

    Args:
        roof_polygons (list): list of roof polygons
        properties (list): list of property

    Returns:
        list: list of footprint polygons
    """
    footprint_polygons = []
    for idx, (roof_polygon, single_property) in enumerate(zip(roof_polygons, properties)):
        if roof_polygon.geom_type == 'MultiPolygon':
            for roof_polygon_ in roof_polygon:
                if roof_polygon_.area < 5:
                    continue
                else:
                    xoffset, yoffset = single_property['xoffset'], single_property['yoffset']
                    transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
                    footprint_polygon = affinity.affine_transform(roof_polygon_, transform_matrix)
                    footprint_polygon = bstool.mask2polygon(bstool.polygon2mask(footprint_polygon))
                    
                    footprint_polygons.append(footprint_polygon)
        elif roof_polygon.geom_type == 'Polygon':
            xoffset, yoffset = single_property['xoffset'], single_property['yoffset']
            transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
            footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)
            footprint_polygon = bstool.mask2polygon(bstool.polygon2mask(footprint_polygon))
             
            footprint_polygons.append(footprint_polygon)
        else:
            continue
            print("Runtime Warming: This processing do not support {}".format(type(roof_polygon)))

    return footprint_polygons

def roof2footprint_single(roof_polygon, offset, offset_model='roof2footprint'):
    """transform roof to footprint

    Args:
        roof_polygon (Polygon): shapely.Polygon
        offset (list): offset
        offset_model (str): 'roof2footprint' or 'footprint2roof

    Returns:
        [type]: [description]
    """
    xoffset, yoffset = offset
    if offset_model == 'roof2footprint':
        transform_matrix = [1, 0, 0, 1, xoffset, yoffset]
    elif offset_model == 'footprint2roof':
        transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
    else:
        raise NotImplementedError

    footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

    return footprint_polygon

def footprint2roof_single(footprint_polygon, offset, offset_model='roof2footprint'):
    """transform footprint to roof

    Args:
        footprint_polygon (Polygon): shapely.Polygon
        offset (list): offset
        offset_model (str): 'roof2footprint' or 'footprint2roof

    Returns:
        [type]: [description]
    """
    xoffset, yoffset = offset
    if offset_model == 'roof2footprint':
        transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
    elif offset_model == 'footprint2roof':
        transform_matrix = [1, 0, 0, 1, xoffset, yoffset]
    else:
        raise NotImplementedError
    try:
        roof_polygon = affinity.affine_transform(footprint_polygon, transform_matrix)
    except:
        print("footprint2roof: ", offset, type(xoffset), type(yoffset), type(footprint_polygon), footprint_polygon)
        roof_polygon = footprint_polygon
    return roof_polygon

def mask_flip(masks, transform_flag='h', image_size=(1024, 1024)):
    """flip the mask

    Args:
        masks (list): list of mask
        transform_flag (str, optional): flag of flip direction. Defaults to 'h'.
        image_size (tuple, optional): size of image. Defaults to (1024, 1024).

    Raises:
        NotImplementedError: [description]

    Returns:
        list: list of masks
    """
    img_height, img_width = image_size
    results = []
    for mask in masks:
        mask = np.array(mask)
        if transform_flag == 'h':
            mask[0::2] = img_width - mask[0::2]
        elif transform_flag == 'v':
            mask[1::2] = img_height - mask[1::2]
        else:
            raise NotImplementedError

        results.append(mask.tolist())

    return results

def mask_rotate(masks, angle=0, center=(512, 512)):
    """rotate mask

    Args:
        masks (list): list of masks
        angle (int, optional): rotation angle. Defaults to 0.
        center (tuple, optional): rotation center. Defaults to (512, 512).

    Returns:
        list: list of masks
    """
    masks = [bstool.polygon2mask(affinity.rotate(bstool.mask2polygon(mask), angle, origin=center, use_radians=True)) for mask in masks]

    return masks