import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def analPctChange(finput,days):
    # read source
    df = pd.read_csv(finput)
    pct_values = df['Adj Close'].pct_change(5)
    print("total data is {}".format(pct_values.count()))
    # figuring out the max and min value
    print("max value is {}".format(pct_values.max()*100))
    print("min value is {}".format(pct_values.min()*100))

    # count based on source
    print("num for <= -0.017 is {}".format(pct_values[pct_values <= -0.017].count()))
    print("num for > -0.017 and <= -0.005 is {}".format(pct_values[(pct_values > -0.017) & (pct_values <= -0.005) ].count()))
    print("num for > -0.005 and <= 0.0035 is {}".format(pct_values[(pct_values > -0.005) & (pct_values <= 0.0030) ].count()))
    print("num for > 0.0035 and <= 0.01 is {}".format(pct_values[(pct_values > 0.0030) & (pct_values <= 0.01) ].count()))
    print("num for > 0.01 and <= 0.02 is {}".format(pct_values[(pct_values > 0.01) & (pct_values <= 0.02) ].count()))
    print("num for > 0.02 is {}".format(pct_values[(pct_values > 0.02)].count()))
    # plot me
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pct_values)
    plt.show()
    #simple_ret = df['Adj Close'].pct_change(days)
    #log_ret = np.log(1+simple_ret)
    #print(np.exp(log_ret.cumsum()[-1])-1)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='a csv file of stock data', required=True)
    parser.add_argument('-n', '--num_of_days',
                        help='num of sequence length', default=5)
    args = parser.parse_args()
    analPctChange(args.input, args.num_of_days)

if __name__ == '__main__':
    main()
