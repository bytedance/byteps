import sys
import os
import numpy as np


folder = sys.argv[1]
keyword = "Training speed"

files = os.listdir(folder)

for filename in files:
    path = os.path.join(folder, filename)
    print(filename)
    speeds = []
    with open(path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if keyword in line:
                print(line.strip('\n'))
                speed = line.split()[-2]
                speeds.append(float(speed))

        speeds = np.array(speeds)
        print("avg: {:.3f}\t std: {:.3f}".format(np.mean(speeds), np.std(speeds)))
