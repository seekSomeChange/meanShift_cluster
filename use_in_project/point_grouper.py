import sys
import numpy as np
import mean_shift_utils1 as mst

GROUP_DISTANCE_TOLERANCE  = 1

class PointGrouper(object):
    def group_points(self,points):
        group_assignment = []
        groups =[]
        group_index = 0
        for point in points:
            neareast_group_index = self._determine_nearest_group(point,groups)
            if neareast_group_index is None:
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(neareast_group_index)
                groups[neareast_group_index].append(point)
        return np.array(group_assignment)



    def _distance_to_group(self,point,group):
        min_distance = sys.float_info.max
        for pt in group:
            dist = mst.lon_lat_distance(point,pt)
            if dist<min_distance:
                min_distance = dist
        return min_distance

    def _determine_nearest_group(self,point,groups):
        nearest_group_index = None
        index = 0
        for group in groups:
            distance_to_group = self._distance_to_group(point,group)
            if distance_to_group<GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
        return nearest_group_index

