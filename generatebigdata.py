import os
from shutil import copyfile


def cre8outputdir(pathdir):
    # create folder output
    if not os.path.exists("{}/bigdata".format(pathdir)):
        os.mkdir("{}/bigdata".format(pathdir))

    if not os.path.exists("{}/bigdata/train".format(pathdir)):
        os.mkdir("{}/bigdata/train".format(pathdir))

    if not os.path.exists("{}/bigdata/test".format(pathdir)):
        os.mkdir("{}/bigdata/test".format(pathdir))

    if not os.path.exists("{}/bigdata/train/0".format(pathdir)):
        os.mkdir("{}/bigdata/train/0".format(pathdir))

    if not os.path.exists("{}/bigdata/train/1".format(pathdir)):
        os.mkdir("{}/bigdata/train/1".format(pathdir))

    if not os.path.exists("{}/bigdata/test/0".format(pathdir)):
        os.mkdir("{}/bigdata/test/0".format(pathdir))

    if not os.path.exists("{}/bigdata/test/1".format(pathdir)):
        os.mkdir("{}/bigdata/test/1".format(pathdir))


pathdir = "dataset"

cre8outputdir(pathdir)

counttest = 0
counttrain = 0
for root, dirs, files in os.walk(pathdir):
    for file in files:
        if file[0] == '0':
            if 'test' in file:
                origin = "{}/{}".format(root, file)
                destination = "{}/bigdata/test/0/{}".format(pathdir, file)
                copyfile(origin, destination)
                counttest += 1
            elif 'train' in file:
                origin = "{}/{}".format(root, file)
                destination = "{}/bigdata/train/0/{}".format(pathdir, file)
                copyfile(origin, destination)
                counttrain += 1
        elif file[0] == '1':
            if 'test' in file:
                origin = "{}/{}".format(root, file)
                destination = "{}/bigdata/test/1/{}".format(pathdir, file)
                copyfile(origin, destination)
                counttest += 1
            elif 'train' in file:
                origin = "{}/{}".format(root, file)
                destination = "{}/bigdata/train/1/{}".format(pathdir, file)
                copyfile(origin, destination)
                counttrain += 1

print(counttest)
print(counttrain)
