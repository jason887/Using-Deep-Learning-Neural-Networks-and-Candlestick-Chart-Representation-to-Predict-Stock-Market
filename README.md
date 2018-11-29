# Stock-Market-Predcition-using-ResNet

## 1. Prepare Environment
run init.sh to create a virtual python environment and install the dependencies

```
$ bash init.sh
```
if you don't want use virtual environment, you can install requirement libraries with :
```
$ pip install -r requirements.txt
```
Highly recomended using virtual environment.

## 2. Prepare dataset
To download data, we provide 2 source, yahoo and tiingo (yahoo by default). We can read a list of stock market and run it. Example, we want to download and preprocess all stock market in tw50.csv with 20 period days and produce 50x50 image dimension.

```
$ python runallfromlist.py tw50.csv 20 50
```
Generate the final dataset. Example, we want to generate a final dataset from tw50 with 20 period days and 50 dimension.
```
$ python generatebigdata.py dataset 20_50 bigdata_20_50
```

## 3. Build the model
We can run build model with default parameter.
```
$ python myDeepCNN.py -i dataset/bigdata_20_50
```
