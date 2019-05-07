#!/bin/python
#coding:UTF-8
'''
Date:20190402
@author: sylvia
'''

import math
from math import *
import sys
import numpy as np
import matplotlib.pyplot as plt

MIN_DISTANCE = 0.01#mini error
RADIUS = 1

def load_data(path,dim):
    '''
    :param filename:
    :return:data
    '''
    f = open(path,'r')
    data = []

    for line in f.readlines()[1:]:
        lines = line.strip().split(',')
        data_tmp = []

        if (len(lines) != dim):
            continue
        for i in xrange(dim):
            if i>0:
                data_tmp.append(float(lines[i]))
            else:
                data_tmp.append(lines[i])

        data.append(data_tmp)
    return data


def write_to_file(filename,users,points,shift_points,cluster):

    f = open(filename,'w')
    ind = 0
    for i in xrange(len(cluster)):
        for j in xrange(len(cluster[i])):
            f.write(("%s\t%5.2f,%5.2f\t%5.2f,%5.2f\t%i\n") % (users[ind],points[i][j][0], points[i][j][1], shift_points[i][j][0], shift_points[i][j][1], cluster[i][j]))
            ind += 1
    f.close()


def lon_lat_distance(pointA,pointB):
    if (len(pointA) != len(pointB)):
        print('Dimensions of A and B should be the same!')
    radius = 6378.137
    a = pointA[0,0] * acos(-1) / 180 - pointB[0,0] * acos(-1) / 180
    b = pointA[0,1] * acos(-1) / 180 - pointB[0,1] * acos(-1) / 180
    s = 2 * asin(sqrt((sin(a / 2) ** 2) + cos(pointA[0,1] * acos(-1) / 180) * cos(pointB[0,1] * acos(-1) / 180) * (sin(b / 2)** 2)))
    s = float(radius * s)

    return s

def gaussian_kernel(distance, bandwidth):
    m = np.shape(distance)[0]
    right = np.mat(np.zeros((m, 1)))
    for i in xrange(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))

    gaussian_val = left * right
    return gaussian_val

def shift_point(point, points, kernel_bandwidth):
    points = np.mat(points)
    m,n = np.shape(points)
    #计算距离
    point_distances = np.mat(np.zeros((m,1)))

    for i in xrange(m):
        # point_distances[i, 0] = np.sqrt((point - points[i]) * (point - points[i]).T)
        if ((point[0] == points[i]).all()):
            continue
        point_distances[i, 0] = lon_lat_distance(point,points[i])

    #计算高斯核
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)

    #计算分母
    all = 0.0
    for i in xrange(m):
        all += point_weights[i, 0]

    #均值偏移
    point_shifted = point_weights.T * points / all
    return point_shifted


def euclidean_dist(pointA, pointB):
    #计算pointA和pointB之间的欧式距离
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)

def distance_to_group(point, group):
    min_distance = 10000.0
    for pt in group:
        dist = euclidean_dist(point, pt)
        if dist < min_distance:
            min_distance = dist
    return min_distance

def group_points(mean_shift_points):
    group_assignment = []
    m,n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in xrange(m):
        item = []
        for j in xrange(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        print item_1
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1

    for i in xrange(m):
        item = []
        for j in xrange(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])

    return group_assignment

def train_mean_shift(points, kernel_bandwidth=2):
    #shift_points = np.array(points)

    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iter = 0
    m, n = np.shape(mean_shift_points)
    need_shift = [True] * m

    #cal the mean shift vector
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iter += 1
        print "iter : " + str(iter)
        for i in range(0, m):
            #判断每一个样本点是否需要计算偏置均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kernel_bandwidth)
            dist = lon_lat_distance(p_new, p_new_start)

            if dist > max_min_dist:#record the max in all points
                max_min_dist = dist
            if dist < MIN_DISTANCE:#no need to move
                need_shift[i] = False

            mean_shift_points[i] = p_new
    #计算最终的group
    group = group_points(mean_shift_points)

    return np.mat(points), mean_shift_points, group

def plot_result_figure(points,shift_points,cluster,n):
    # f = open(filename)
    label_seq = []
    cluster_x = []
    cluster_y = []
    center_x = []
    center_y = []
    center_label = {}

    for i in xrange(len(points)):
        # lines = line.strip().split("\t")
        # if len(lines) == 3:
        label = int(cluster[i])
        label_seq.append(label)
        data_1 = points[i]
        cluster_x.append(float(data_1[0]))
        cluster_y.append(float(data_1[1]))
        data_2 = shift_points[i]

        if label not in center_label.keys():
            center_label[label] = label
            center_x.append(float(data_2[0]))
            center_y.append(float(data_2[1]))
    # f.close()

    # plt.plot(cluster_x_0, cluster_y_0, 'b.', label="cluster_0")
    # plt.plot(cluster_x_1, cluster_y_1, 'g.', label="cluster_1")
    # plt.plot(cluster_x_2, cluster_y_2, 'k.', label="cluster_2")
    # plt.plot(center_x, center_y, 'r+', label="mean point")
    # plt.title('Mean Shift 2')
    # #plt.legend(loc="best")
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label in center_label.keys():
        ind = [i for i in range(0,len(label_seq)) if label_seq[i]==label ]
        cluster_name ='cluster_'+str(label)
        x = [cluster_x[i] for i in ind]
        y = [cluster_y[i] for i in ind]

        ax.scatter(x,y,marker='.',label = cluster_name)

    ax.scatter(center_x,center_y,c = 'red',marker='+',label = 'mean point',s=80)
    plt.legend()

    fig.savefig("user_" + str(n))

def run():
    # 导入数据集
    data = load_data('data.csv',3)

    # 遍历每个用户的夜间轨迹点
    ind = 0
    lon_lat = []
    lon_lat.append([])
    user_id = data[0][0]
    users = []

    for d in xrange(len(data)):
        user = data[d][0]
        users.append(user)
        if (user == user_id):
            lon_lat[ind].append(data[d][-1:-3:-1])
        else:
            ind += 1
            user_id = user
            lon_lat.append([])

    users=np.array(users)
    #训练，h=2
    j=0
    points =[]
    shift_points = []
    cluster = []
    for i in xrange(len(lon_lat)):
        tmp_points, tmp_shift_points, tmp_cluster = train_mean_shift(lon_lat[i], 2)
        tmp_points = np.array(tmp_points)
        tmp_shift_points=np.array(tmp_shift_points)
        tmp_cluster = np.array(tmp_cluster)

        points.append(tmp_points)
        shift_points.append(tmp_shift_points)
        cluster.append(tmp_cluster)
        plot_result_figure(tmp_points,tmp_shift_points,tmp_cluster, i)
        del tmp_cluster,tmp_shift_points,tmp_points



    filename = 'output.txt'
    write_to_file(filename,users,points,shift_points,cluster)


    #  绘制聚类结果图像

# -------------------------------------------------------------------------

if __name__ == "__main__":
    run()
