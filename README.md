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

### Convert O-L-H-C stock data into image
```shell-script
$ python preprocessing.py -m convert2image -i dataset/images/ -l 20
```
