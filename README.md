# meanShift_cluster
use mean shift method to identify cellphone user's residence in night 

input data format:
file data:userid,lat,lon
file cz_user:userid

output:mean shift results for resident clusters' centers of different permanent residents.

reference link:https://blog.csdn.net/google19890102/article/details/51030884

## Problem description:
There is some users trace points data,now I want to identify each user's high-frequency activity areas.To solve this problem,I chose mean shift clustering method which is based on points' density.More informations about this method could find in the reference link.In this project,I used Latitude and longitude spherical distance instead of eucilidian distance.

### The trace points cluster results of two test users are showed in figures as:
![user1](https://github.com/seekSomeChange/meanShift_cluster/blob/ba181a636d21e30722e3fa7ef967fe12a3037177/user_0.png)
![user2](https://github.com/seekSomeChange/meanShift_cluster/blob/ba181a636d21e30722e3fa7ef967fe12a3037177/user_1.png)

