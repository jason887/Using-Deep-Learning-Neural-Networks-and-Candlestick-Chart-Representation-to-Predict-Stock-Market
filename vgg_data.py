import os
import shutil
import random
import errno
import argparse
from distutils.dir_util import copy_tree

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-r', '--ratio',
                        help='ratio for split data', type=float, default=0.2)
    args = parser.parse_args()

    build_vgg_data(args.input, args.ratio)

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def build_vgg_data(input, ratio):
    root_path = os.getcwd()
    path = input.split('/')
    length = path[1]
    source_path = "{}/{}".format(root_path,input)
    print("source path : {}".format(source_path))
    new_path = "{}/{}/vgg_{}".format(root_path,path[0], length)
    print("new path : {}".format(new_path))
    train_folder = "{}/training".format(new_path)
    validation_folder = "{}/validation".format(new_path)
    print("train folder : {}".format(train_folder))
    print("validation folder : {}".format(validation_folder))
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)
    # copy_tree(source_path, train_folder)
    num_files_train_A = sum([len(files) for r, d, files in os.walk("{}/A".format(train_folder))])
    print("numoffilesA : {}".format(num_files_train_A))
    for i

if __name__ == "__main__":
    main()
