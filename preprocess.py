import pandas as pd
import plotly.offline as offline
import matplotlib.pyplot as plt
from plotly.tools import FigureFactory as FF
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# offline.init_notebook_mode()

import argparse

import decimal

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
                        help='an csv file of stock data', required=True)
    parser.add_argument('-l', '--seq_len',
                        help='num of sequence length', default=20)
    parser.add_argument('-m', '--mode',
                        help='mode of preprocessing data', required=True)
    args = parser.parse_args()
    if args.mode == 'convert2image':
        convert2image(args.input,args.seq_len)
    if args.mode == 'createLabel':
        createLabel(args.input, args.seq_len)
    if args.mode == 'img2dt':
        image2dataset(args.input)

def image2dataset(input):
    img = load_img(input)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    print(x.shape)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    #print(x)

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
        #if perct in range(-1.5, -0.5):
            label = "B"
        if perct > -0.5 and perct < 0.4:
        #if perct in range(-0.5, 0.4):
            label = "C"
        if perct > 0.4 and perct < 1.4:
        #if perct in range(0.4, 1.4):
            label = "D"
        if perct > 1.4 and perct < 2.5:
        #if perct in range(1.4, 2.5):
            label = "E"
        if perct > 2.5 and perct < 4.3:
        #if perct in range(2.5, 4.3):
            label = "F"
        if perct > 4.3:
            label = "G"
        # print("{},{}-{}".format(perct, fname[12:-4], i))
        with open("{}_label.txt".format(fname[12:-4]), 'a') as the_file:
            the_file.write("{},{}-{}\n".format(label, fname[12:-4], i))


def convert2image(fname, seq_len):
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
            'paper_bgcolor': 'rgba(1,1,1,1)',
            'plot_bgcolor': 'rgba(1,1,1,1)'
        })
        #plot_mpl(fig, image='png')
        #py.image.save_as(fig, filename='dataset/images/{}.png'.format(i))
        offline.plot(fig, filename='dataset/images/{}-{}.html'.format(fname[12:-4], i),
                     image='png', auto_open=False, show_link=False, image_filename='dataset/images/{}-{}.png'.format(fname[11:-4], i))
    # imagemagic script to resize img
    # find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -adaptive-resize 48x48! "{}" "{}"
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
