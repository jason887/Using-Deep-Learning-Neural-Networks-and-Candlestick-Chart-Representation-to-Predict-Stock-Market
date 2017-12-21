import pandas as pd
import plotly.offline as offline
import matplotlib.pyplot as plt
from plotly.tools import FigureFactory as FF
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# offline.init_notebook_mode()
import glob
import argparse
import os
import decimal
from shutil import copyfile

import imgkit

import subprocess

def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)


def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='a csv file of stock data', required=True)
    parser.add_argument('-l', '--seq_len',
                        help='num of sequence length', default=20)
    parser.add_argument('-lf', '--label_file',
                        help='a label_file')
    parser.add_argument('-d', '--dimension',
                        help='a dimension value')
    parser.add_argument('-t', '--dataset_type',
                        help='training or testing datasets')
    parser.add_argument('-m', '--mode',
                        help='mode of preprocessing data', required=True)
    args = parser.parse_args()
    if args.mode == 'olhc2cs':
        olhc2cs(args.input, args.seq_len, args.dataset_type)
    if args.mode == 'createLabel':
        createLabel(args.input, args.seq_len)
    if args.mode == 'img2dt':
        image2dataset(args.input, args.label_file)
    if args.mode == 'countImg':
        countImage(args.input)
    if args.mode == 'html2img':
        html2img(args.input, args.dimension)

def html2img(input, dim):
    # get a list of all the files to open
    glob_folder = os.path.join("{}/html".format(input), '*.html')
    path_html = "{}/{}/html".format(os.getcwd(),input)
    path_img = "{}/{}/img".format(os.getcwd(),input)
    html_file_list = glob.glob(glob_folder)
    index = 1
    os.chdir(path_img)
    for html_file in html_file_list:
        # print("html name : {}".format(html_file))
        filename = html_file.split('/')
        # get the name into the right format
        temp_name = "{}/{}".format(path_html, filename[4])
        # print("temp_name : {}".format(temp_name))

        # print("html name : {}".format(filename))
        pngfile = "{}/{}.png".format(path_img, filename[4][:-5])

        print("convert {} to {}".format(temp_name,pngfile))
        imgkit.from_file(temp_name, pngfile)
    # crop only take the content
    # subprocess.call('find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -crop 700x458+0+0 "{}" "{}"', shell=True)
    imgsize = "{}x{}!".format(dim,dim)
    subprocess.call('find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -flatten +matte -adaptive-resize '+ str(imgsize) +' "{}" "{}"', shell=True)
    # subprocess.call('convert rgb10.png -pointsize 50 -draw "text 180,180 ' + str(tempo) + '" rgb10-n.png', shell=True)

def image2dataset(input, label_file):

    label_dict = {}
    with open(label_file) as f:
        for line in f:
            (key, val) = line.split(',')
            label_dict[key] = val.rstrip()
    # print(label_dict)
    # print(list(label_dict.values())[list(label_dict.keys()).index('FTSE-80')])
    path = "{}/{}".format(os.getcwd(), input)
    # df = pd.DataFrame()
    # os.chdir("{}/{}/".format(os.getcwd(),input))
    # print(os.getcwd())

    for filename in os.listdir(input):
        print(filename)
        print(os.getcwd())
        if filename is not '':
            label = list(label_dict.values())[
                list(label_dict.keys()).index("{}".format(filename[:-4]))]
            # name = list(label_dict.keys())[list(label_dict.values()).index("{}".format(label))]
            #print("name : {}".format(name))
            # print(filename)
            new_name = "{}{}.png".format(label, filename[:-4])
            print("rename {} to {}".format(filename, new_name))
            os.rename("{}/{}".format(path,filename), "{}/{}".format(path,new_name))

    folders = ['A','B','C','D','E','F','G']
    for folder in folders:
        if not os.path.exists("{}/classes/{}".format(path,folder)):
            os.makedirs("{}/classes/{}".format(path,folder))

    for filename in os.listdir(input):
        if filename is not '':
            # print(filename[:1])
            if filename[:1] == "A":
                copyfile("{}/{}".format(path,filename), "{}/classes/A/{}".format(path,filename))
            elif filename[:1] == "B":
                copyfile("{}/{}".format(path,filename), "{}/classes/B/{}".format(path,filename))
            elif filename[:1] == "C":
                copyfile("{}/{}".format(path,filename), "{}/classes/C/{}".format(path,filename))
            elif filename[:1] == "D":
                copyfile("{}/{}".format(path,filename), "{}/classes/D/{}".format(path,filename))
            elif filename[:1] == "E":
                copyfile("{}/{}".format(path,filename), "{}/classes/E/{}".format(path,filename))
            elif filename[:1] == "F":
                copyfile("{}/{}".format(path,filename), "{}/classes/F/{}".format(path,filename))
            elif filename[:1] == "G":
                copyfile("{}/{}".format(path,filename), "{}/classes/G/{}".format(path,filename))


def createLabel(fname, seq_len):
    # import plotly.graph_objs as go
    #py.sign_in('rosdyana', 'eVtlDykeB8gMHmp6y4Ff')
    # read stock data
    df = pd.read_csv(fname, header=None, index_col=0)
    df = df[4]
    df.fillna(0)
    # df = df.astype(str)
    # separators = pd.DataFrame(', ', df.index, df.columns[:-1])
    # separators[df.columns[-1]] = '\n'
    # print (df + separators).sum(axis=1).sum()

    data = df[1:]
    # print(data)

    for i in range(0, len(data), int(seq_len)):
        #print("idx : {}".format(i))
        c = data[i:i + int(seq_len)]
        # print(len(c))
        starting = 0
        endvalue = 0
        label = ""
        for idx, val in enumerate(c):
            if idx == 0:
                starting = float(val)
            if idx == len(c) - 1:
                endvalue = float(val)
        sizeincrease = endvalue - starting
        diff = sizeincrease / starting
        perct = diff * 100
        if isnan(perct):
            perct = 0
        if perct < -1.5:
            label = "A"
        if perct > -1.5 and perct < -0.5:
            # if perct in range(-1.5, -0.5):
            label = "B"
        if perct > -0.5 and perct < 0.4:
            # if perct in range(-0.5, 0.4):
            label = "C"
        if perct > 0.4 and perct < 1.4:
            # if perct in range(0.4, 1.4):
            label = "D"
        if perct > 1.4 and perct < 2.5:
            # if perct in range(1.4, 2.5):
            label = "E"
        if perct > 2.5 and perct < 4.3:
            # if perct in range(2.5, 4.3):
            label = "F"
        if perct > 4.3:
            label = "G"
        # print("{},{}-{}".format(perct, fname[12:-4], i))
        with open("{}_label_{}.txt".format(fname[12:-4],seq_len), 'a') as the_file:
            the_file.write("{}-{},{}".format(fname[12:-4], i, label))
            the_file.write("\n")


def countImage(input):
    num_file = sum([len(files) for r, d, files in os.walk(input)])
    num_dir = sum([len(d) for r, d, files in os.walk(input)])
    print("num of files : {}\nnum of dir : {}".format(num_file, num_dir))


def olhc2cs(fname, seq_len, dataset_type):
    print("Converting olhc to candlestick")
    path = "{}".format(os.getcwd())
    print(path)
    if not os.path.exists("{}/dataset/{}/{}/html/".format(path,seq_len),dataset_type):
        os.makedirs("{}/dataset/{}/{}/html/".format(path,seq_len,dataset_type))
        os.makedirs("{}/dataset/{}/{}/img/".format(path,seq_len,dataset_type))
    # import plotly.graph_objs as go
    #py.sign_in('rosdyana', 'eVtlDykeB8gMHmp6y4Ff')
    # read stock data
    df = pd.read_csv(fname, header=None, index_col=0)
    df.fillna(0)
    # drop date and volume columns
    df.drop(df.columns[[4, 5]], axis=1, inplace=True)
    df = df.astype(str)
    separators = pd.DataFrame(', ', df.index, df.columns[:-1])
    separators[df.columns[-1]] = '\n'
    # print (df + separators).sum(axis=1).sum()
    data = df[1:]
    # print(data.head())

    for i in range(0, len(data), int(seq_len)):
        print("idx : {}".format(i))
        c = data[i:i + int(seq_len)]
        fig = FF.create_candlestick(
            open=c[1], high=c[2], low=c[3], close=c[4])

        fig['layout'].update({
            'xaxis': dict(visible=False),
            'yaxis': dict(visible=False),
            'paper_bgcolor': 'rgb(255,255,255)',
            'plot_bgcolor': 'rgb(255,255,255)'
        })
        #plot_mpl(fig, image='png')
        #py.image.save_as(fig, filename='dataset/images/{}.png'.format(i))
        offline.plot(fig, filename='dataset/{}/{}/html/{}-{}.html'.format(seq_len,dataset_typename[12:-4], i),
                     image='png', auto_open=False, show_link=False, image_filename='{}-{}.png'.format(fname[11:-4], i))


    # imagemagic script to resize img
    # find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -flatten +matte -adaptive-resize 48x48! "{}" "{}"
    # R Script convert html to img
    # library(webshot)
    # html_files <- list.files(pattern = ".html$", recursive = TRUE)
    # for(i in html_files){
    #   webshot(i, sprintf("%s", paste(i, "png", sep=".")),delay = 0.5)
    #   #print(sprintf("%s", paste(i, "png", sep=".")))
    #   print("done")
    # }
if __name__ == '__main__':
    main()
