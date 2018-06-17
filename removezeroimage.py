import os
import sys
from PIL import Image

pathdir = sys.argv[1]
origindir = sys.argv[2]

countme = 0
# dir walk to get all images file
for root, dirs, files in os.walk("{}/{}".format(pathdir, origindir)):
    for file in files:
        pathimg = "{}/{}".format(root, file)
        img = Image.open(pathimg)
        xx = img.mode
        if not xx == 'P':
            os.remove(pathimg)
            countme += 1

print(countme)
