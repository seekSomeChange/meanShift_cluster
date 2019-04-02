# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mean_shift as ms


def load_data(filename):
    '''
    :param filename:
    :return:data
    '''
    data = np.genfromtxt(filename,delimiter=',',skip_header=1,dtype = None)
    return data

def load_points(datafile,userfile):
    '''
    :param datafile,userfile
    :return: points for each user
    notes:msids' orders in two files should be match
    '''
    data = load_data(datafile)
    users = load_data(userfile)

    ind = 0
    points_lon = []
    points_lat = []
    points_lon.append([])
    points_lat.append([])
    for d in range(0,data.shape[0]):
        if data[d][0]==users[ind]:
            points_lat[ind].append(data[d][1])
            points_lon[ind].append(data[d][2])
        elif ind<len(users):
            ind += 1
            points_lon.append([])
            points_lat.append([])

    return points_lon,points_lat

# def write_to_file():


def run():
    # user_files = load_data('user_files.csv')
    # user_number=0

    # user_points_lon,user_points_lat = load_points('data.csv','cz_msid.csv')
    #
    # for i in range(0,len(user_points_lon)):
    #     points = [user_points_lon[i],user_points_lat[i]]
    #     mean_shifter = ms.MeanShift()
    #     mean_shift_result = mean_shifter.cluster(points,kernel_bandwidth=3)
    #     print('user_number:%d\n') % i
    #     print "Original Point     Shifted Point  Cluster ID"
    #     print "============================================"
    #     for j in range(len(mean_shift_result.shifted_points)):
    #         original_point = mean_shift_result.original_points[j]
    #         converged_point = mean_shift_result.shifted_points[j]
    #         cluster_assignment = mean_shift_result.cluster_ids[j]
    #         print "(%5.2f,%5.2f)  ->  (%5.2f,%5.2f)  cluster %i" % (original_point[0], original_point[1], converged_point[0], converged_point[1], cluster_assignment)

    # ------------------------------------------------------------------------

    user_points_lon, user_points_lat = load_points('data.csv', 'cz_msid.csv')

    for k in range(0,len(user_points_lon)):
        points = [user_points_lon[k],user_points_lat[k]]
        points = np.array(points).T
        mean_shifter = ms.MeanShift()
        mean_shift_result = mean_shifter.cluster(points, kernel_bandwidth=15)

        original_points = mean_shift_result.original_points
        shifted_points = mean_shift_result.shifted_points
        cluster_assignments = mean_shift_result.cluster_ids

        x = original_points[:, 0]
        y = original_points[:, 1]
        Cluster = cluster_assignments
        centers = shifted_points

        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(x, y, c=Cluster, s=50)
        for i, j in centers:
            ax.scatter(i, j, s=50, c='red', marker='+')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(scatter)

        fig.savefig("mean_shift_result_"+str(k))


if __name__ == '__main__':
    run()




