from __future__ import print_function
import os
import csv
import sys
import numpy as np
from collections import defaultdict

if __name__ == '__main__':
  lFile = sys.argv[1]
  f = open(lFile, 'r')
  lines = [line.rstrip('\n') for line in f]

  trackDict = defaultdict(list)
  trackDict['det_kp_per_img'] = defaultdict(list)
  trackDict['det_kp_per_img_all'] = defaultdict(list)
  trackDict['det_perf_per_img_all'] = defaultdict(list)  
  trackDict['det_desc_perf_per_img_all'] = defaultdict(list)  
  trackDict['matches_per_img'] = defaultdict(list)

  for line in lines:
    keys = line.rsplit(' ')
    keys0 = keys[0].rsplit('-')

    if keys[1] == 'detection':
        if keys[0] not in trackDict['det_kp_per_img_all']:
            trackDict['det_kp_per_img_all'][keys[0]] = defaultdict(list)
            trackDict['det_perf_per_img_all'][keys[0]] = defaultdict(list)
        trackDict['det_kp_per_img_all'][keys[0]][keys[7]].append(int(keys[3]))
        trackDict['det_perf_per_img_all'][keys[0]][keys[7]].append(float(keys[9]))

    if keys[1] == 'descriptor':
        if keys[0] not in trackDict['det_desc_perf_per_img_all']:
            trackDict['det_desc_perf_per_img_all'][keys[0]] = defaultdict(list)

        dkey = (keys[0].rsplit('-'))[0]
        det_perf = trackDict['det_perf_per_img_all'][dkey][keys[5]][-1]
        trackDict['det_desc_perf_per_img_all'][keys[0]][keys[5]].append(float(keys[7])+det_perf)


    if (len(keys) > 4 ) and (keys[4] == 'matches'):
        trackDict['matches_per_img'][keys[0]].append(int(keys[2]))

  for key in trackDict['det_kp_per_img_all']:
      frames = [trackDict['det_kp_per_img_all'][key][frm][0] for frm in trackDict['det_kp_per_img_all'][key]]
      trackDict['det_kp_per_img'][key].append(frames)
  f.close()

  print('MP.7 Performance Evaluation 1\n')
  print('Detector | #Total Keypoints | Per Img KPs')
  print('------------ | ------------- | -------------')

  for key in trackDict['det_kp_per_img']:
      frames = trackDict['det_kp_per_img'][key][0]
      print('{:10s} | {:5d} | {}'.format(key, int(np.sum(frames)), frames))

  print('\nMP.8 Performance Evaluation 2\n')
  print('{:35s} | {} | {}'.format('Det-Desc-Mat-Sel', '#Total Matches', 'Per Img Matches'))
  print('------------ | ------------- | -------------')
  for key in trackDict['matches_per_img']:
      mkey = key.split('-')
      if mkey[2]=='MAT_BF' and mkey[3]=='SEL_KNN':
          frames = trackDict['matches_per_img'][key]
          print('{:35s} | {:4d} | {}'.format(key, int(np.sum(frames)), frames))

  print('\nMP.9 Performance Evaluation 3\n')
  perf_f = open('results/Performance.csv', mode='w', newline='')
  perf_w = csv.writer(perf_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  perf_w.writerow(['Detector-Descriptor', 'Total Keypoints in 10 Images', 'Avg Perf Per Image(ms)', 'Avg Perf Per KeyPoint(us)'])

  for key in trackDict['det_desc_perf_per_img_all']:
      dkey = (key.rsplit('-'))[0]
      frames = trackDict['det_kp_per_img'][dkey][0]
      frames_perf = [np.mean(trackDict['det_desc_perf_per_img_all'][key][frm]) for frm in trackDict['det_desc_perf_per_img_all'][key]]
      perf_w.writerow([key, int(np.sum(frames)), np.mean(frames_perf), np.mean(frames_perf)/np.sum(frames)])

  perf_f.close()