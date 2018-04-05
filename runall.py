import subprocess
import os

subprocess.call(
    'python resnet50.py -i blackmulti/20/EWT -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/20/EIDO -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/20/FTW -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/20/IDX -d 48 -e 100 -o multiclassresult.txt', shell=True)

subprocess.call(
    'python resnet50.py -i blackmulti/10/EWT -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/10/EIDO -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/10/FTW -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/10/IDX -d 48 -e 100 -o multiclassresult.txt', shell=True)

subprocess.call(
    'python resnet50.py -i blackmulti/5/EWT -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/5/EIDO -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/5/FTW -d 48 -e 100 -o multiclassresult.txt', shell=True)
subprocess.call(
    'python resnet50.py -i blackmulti/5/IDX -d 48 -e 100 -o multiclassresult.txt', shell=True)

# subprocess.call(
#     'python resnet101.py -i blackmulti/20/EWT -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/20/EIDO -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/20/FTW -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/20/IDX -d 48 -e  100 -o multiclassresult.txt', shell=True)

# subprocess.call(
#     'python resnet101.py -i blackmulti/10/EWT -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/10/EIDO -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/10/FTW -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/10/IDX -d 48 -e  100 -o multiclassresult.txt', shell=True)

# subprocess.call(
#     'python resnet101.py -i blackmulti/5/EWT -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/5/EIDO -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/5/FTW -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet101.py -i blackmulti/5/IDX -d 48 -e  100 -o multiclassresult.txt', shell=True)

# subprocess.call(
#     'python resnet152.py -i blackmulti/20/EWT -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/20/EIDO -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/20/FTW -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/20/IDX -d 48 -e  100 -o multiclassresult.txt', shell=True)

# subprocess.call(
#     'python resnet152.py -i blackmulti/10/EWT -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/10/EIDO -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/10/FTW -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/10/IDX -d 48 -e  100 -o multiclassresult.txt', shell=True)

# subprocess.call(
#     'python resnet152.py -i blackmulti/5/EWT -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/5/EIDO -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/5/FTW -d 48 -e  100 -o multiclassresult.txt', shell=True)
# subprocess.call(
#     'python resnet152.py -i blackmulti/5/IDX -d 48 -e  100 -o multiclassresult.txt', shell=True)

os.system(
    'spd-say --voice-type female3 "hahahahahaha" --rate -50 --pitch 50')
