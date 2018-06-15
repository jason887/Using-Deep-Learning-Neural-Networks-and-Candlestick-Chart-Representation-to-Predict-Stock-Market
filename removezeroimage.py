import os
import sys
import scipy.misc

pathdir = sys.argv[1]
origindir = sys.argv[2]

countme = 0
# dir walk to get all images file
for root, dirs, files in os.walk("{}/{}".format(pathdir, origindir)):
    for file in files:
        if file[0] == '0':
            pathimg = "{}/{}".format(root, file)
            img = scipy.misc.imread(pathimg)
            height, width, chan = img.shape
            if not chan == 3:
                os.remove(pathimg)
                countme += 1
print(countme)
