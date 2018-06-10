import pandas as pd
import os

pathdir = "stockdatas"

for root, dirs, files in os.walk(pathdir):
    for file in files:
        data = pd.read_csv("{}/{}".format(pathdir, file))
        if len(data) == 0:
            print("{},{}".format(file, file.split("_")[0]))
