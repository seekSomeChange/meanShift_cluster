# -*- coding:utf-8 -*-
import numpy as np
import mean_shift_utils1 as mst
from mean_shift_utils1 import *
import point_grouper as pg

MIN_DISTANCE=1e-5

class MeanShift(object):
    def __init__(self,kernel = gaussian_kernel):
        self.kernel = kernel

    def cluster(self,points,kernel_bandwidth,iteration_callback=None):
        if iteration_callback:
            iteration_callback(points,0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        still_shifting = [True] * shift_points.shape[0]
        while max_min_dist>MIN_DISTANCE:
            iteration_number += 1
            max_min_dist = 0
            for i in range(0,len(shift_points)):
                if not still_shifting:
                    continue
                point_new = shift_points[i]
                point_new_start = point_new
                point_new = self._shift_point(point_new,points,kernel_bandwidth)
                dist = mst.lon_lat_distance(point_new,point_new_start)

                # why ?
                if dist>max_min_dist:
                    max_min_dist=dist
                if dist<MIN_DISTANCE:
                    shift_points[i]=False
                shift_points[i]=point_new

            # 迭代次数召回值
            if iteration_callback:
                iteration_callback(shift_points,iteration_number)

            point_grouper = pg.PointGrouper()
            group_assignment = point_grouper.group_points(shift_points.tolist())

            return MeanShiftResult(points,shift_points,group_assignment)

    def _shift_point(self,point,points,kernel_bandwidth):
        points = np.array(points)

        point_weights = self.kernel(point - points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        # denominator
        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        return shifted_point





class MeanShiftResult:
    def __init__(self,original_points,shifted_points,cluster_ids):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids






