import pandas as pd

# read stock data
df = pd.read_csv('SPY.csv', header=None, index_col=0)
# drop date and volume columns
df.drop(df.columns[[4, 5]], axis=1, inplace=True)
df = df.astype(str)
separators = pd.DataFrame(', ', df.index, df.columns[:-1])
separators[df.columns[-1]] = '\n'
# print (df + separators).sum(axis=1).sum()
data = df[1:]
# print(data.head())

for i in range(0, len(data), 20):

    c = data[i:i + 20]
    print(c)
    print("\nnext\n")
# resize 224x224 imagemagick
# find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -adaptive-resize 224x224! "{}" "{}"
