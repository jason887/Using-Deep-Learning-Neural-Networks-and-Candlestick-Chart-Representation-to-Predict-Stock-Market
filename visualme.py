import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.finance import *
import matplotlib.dates as mdates
symbollist = ['EWT','IDX','EIDO','FTW']
typelist = ['training','testing']
for i in typelist:
    for j in symbollist:
        finput = "stockdatas/{}_{}.csv".format(j,i)
        df = pd.read_csv(finput, parse_dates=True, index_col=0)
        outname  = finput.split("/")[1][:-4]
        df.fillna(0)
        df.reset_index(inplace=True)
        df['Date2'] = df['Date'].map(mdates.date2num)
        ohlc = zip(df['Date2'], df['Open'], df['High'], df['Low'], df['Close'], df['Volume'])
        my_dpi = 96
        fig = plt.figure(figsize=(1000/my_dpi, 600/my_dpi), dpi=my_dpi)
        ax1 = plt.subplot2grid((1,1), (0,0))
        candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')
        ax1.set_ylabel('{}'.format(outname), size=20)

        ax1.xaxis_date()
        pngfile='Figure_{}.png'.format(outname)
        fig.savefig(pngfile, pad_inches=0, transparent=False)
        plt.close(fig)
