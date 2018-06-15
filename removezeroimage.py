import os
import sys
from PIL import Image
import subprocess

pathdir = sys.argv[1]
origindir = sys.argv[2]

countme = 0
# dir walk to get all images file
for root, dirs, files in os.walk("{}/{}".format(pathdir, origindir)):
    for file in files:
        if file[0] == '0':
            pathimg = "{}/{}".format(root, file)
            im = Image.open(pathimg)
            if not im.mode == "P":
                os.remove(pathimg)
                countme += 1
print(countme)
