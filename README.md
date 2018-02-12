# Stock-Market-Predcition-using-ResNet

## Prepare data
To download data, we provide 2 source, yahoo and tiingo

```shell-script
$ python get_data.py -t SPY -s yahoo
```
```shell-script
$ python get_data.py -t SPY -s tiingo
```
## Preprocessing Data
We provide ready dataset for run this prediction, but if you want to build your own data, please follow this steps.

### Convert O-H-L-C-V stock data into candle stick plot
```shell-script
$ python preprocess.py -m ohlc2cs -l 20 -i stockdatas/ETF_testing.csv -t testing
```

### Create label from stock price data
```shell-script
python preprocess.py -m createLabel -i stockdatas/^FTSE.csv -l 5
```

### Build dataset folder separated each classes
```shell-script
python preprocess.py -m img2dt -i dataset/classes/ -lf ETF_testing_label_5.txt
```

### Build the model
```shell-script
python resnet18.py -i dataset/10/img/classes -d 200 -c 3 -e 5 -b 16 -o adam
```
