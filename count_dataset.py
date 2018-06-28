import os
import sys

pathdir = sys.argv[1]

counttest = 0
counttrain = 0
negtest = 0
postest = 0
negtrain = 0
postrain = 0
for root, dirs, files in os.walk("{}/test".format(pathdir)):
    for file in files:
        counttest += 1
        if file.startswith('0'):
            negtest += 1
        if file.startswith('1'):
            postest += 1

for root, dirs, files in os.walk("{}/train".format(pathdir)):
    for file in files:
        counttrain += 1
        if file.startswith('0'):
            negtrain += 1
        if file.startswith('1'):
            postrain += 1

print("test num : {}\nnegative : {}\npositive : {}\n====\ntrain num : {}\nnegative : {}\npositive : {}".format(
    counttest, negtest, postest, counttrain, negtrain, postrain))
