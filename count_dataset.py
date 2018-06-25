import os
import sys

pathdir = sys.argv[1]

counttest = 0
counttrain = 0
for root, dirs, files in os.walk("{}/test".format(pathdir)):
    for file in files:
        counttest += 1

for root, dirs, files in os.walk("{}/train".format(pathdir)):
    for file in files:
        counttrain += 1

print("test num : {}\ntrain num : {}".format(counttest, counttrain))
