# meanShift_cluster
use mean shift method to identify cellphone user's residence in night 

input data format:
file data:userid,lat,lon
file cz_user:userid

output:mean shift results for resident clusters' centers of different permanent residents.

reference link:https://blog.csdn.net/google19890102/article/details/51030884

Problem description:
There is some users trace points data,now I want to identify each user's high-frequency activity areas.
我有一些用户多天的轨迹点数据，现在想要将每个用户的轨迹点分别进行聚类处理，进而识别出每个用户的高频活动区域。

本次项目中选取了均值聚类方法。该方法是一种基于密度的聚类方法。
To solve this problem,I chose mean shift clustering method which is based on points' density.More informations about this method could find in the reference link.In this project,I used Latitude and longitude spherical distance instead of eucilidian distance.

The trace points cluster results of two test users are showed in figures as:
https://github.com/seekSomeChange/meanShift_cluster/blob/master/use_in_project/user_0.png
https://github.com/seekSomeChange/meanShift_cluster/blob/master/use_in_project/user_1.png
