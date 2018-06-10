import pandas as pd
import sys
import subprocess

fileinput = sys.argv[1]
windowsize = sys.argv[2]
dimension = sys.argv[3]
data = pd.read_csv(fileinput)

for i in data['ticker']:
    task = ["python", "run_binary_preprocessing.py",
            str(i), windowsize, dimension]
    subprocess.call(task)
