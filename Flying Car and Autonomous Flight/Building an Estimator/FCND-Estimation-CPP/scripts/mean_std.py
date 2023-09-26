import numpy as np

if __name__ == "__main__":
    gpsX = np.loadtxt('results/log_mean_std/Graph1.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]
    imuAx = np.loadtxt('results/log_mean_std/Graph2.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]

    gpsX_std = np.std(gpsX)
    imuAx_std = np.std(imuAx)
    print('GPSPosXY_Std - {} with {} samples'.format(gpsX_std, len(gpsX)))
    print('AccelXY_Std - {} with {} samples'.format(imuAx_std, len(imuAx)))