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

### Convert O-L-H-C stock data into candle stick plot
```shell-script
$ python preprocess.py -m olhc2cs -i stockdatas/^FTSE.csv -l 5
```

### Create label from stock price data
```shell-script
python preprocess.py -m createLabel -i stockdatas/^FTSE.csv -l 5
```

### Convert candle stick plot html to img
```shell-script
python preprocess.py -m html2img -i dataset/5 -d 200
```

### Build dataset folder separated each classes
```shell-script
python preprocess.py -m img2dt -i dataset/5/img -lf FTSE_label_5.txt
```

### Build the model
```shell-script
python resnet18.py -i dataset/10/img/classes -d 200 -c 3 -e 5 -b 16 -o adam
```
