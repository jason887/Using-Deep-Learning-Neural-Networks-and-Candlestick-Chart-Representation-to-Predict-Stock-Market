import subprocess
import os

formatters = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'END': '\033[0m',
}

symbol = "EIDO"
start_date = "2017-01-01"
end_date = "2016-12-31"
windows_length = 5

# get data testing
print('{RED}\nGet Testing Data{END}'.format(**formatters))
subprocess.call(
    'python get_data.py -sd {} -t {} -s yahoo -p testing'.format(start_date, symbol), shell=True)
print('{GREEN}Get Testing Data Done\n{END}'.format(**formatters))

# get data testing
print('{RED}\nGet Training Data{END}'.format(**formatters))
subprocess.call(
    'python get_data.py -ed {} -t {} -s yahoo -p training'.format(end_date, symbol), shell=True)
print('{GREEN}Get Training Data Done\n{END}'.format(**formatters))

# create label training
print('{RED}\nCreate Label Training Data{END}'.format(**formatters))
subprocess.call('python preprocess.py -m createLabel -l {} -i stockdatas/{}_training.csv'.format(
    windows_length, symbol), shell=True)
print('{GREEN}Create Label Training Data Done\n{END}!'.format(**formatters))

# create label testing
print('{RED}\nCreate Label Testing Data{END}'.format(**formatters))
subprocess.call('python preprocess.py -m createLabel -l {} -i stockdatas/{}_testing.csv'.format(
    windows_length, symbol), shell=True)
print('{GREEN}Create Label Testing Data Done\n{END}'.format(**formatters))

# convert to candlestick chart training data
print('{RED}\nConvert Training Data to Candlestik{END}'.format(**formatters))
subprocess.call('python preprocess.py -m ohlc2cs -l {} -i stockdatas/{}_training.csv -t training'.format(
    windows_length, symbol), shell=True)
print('{GREEN}Convert Training Data to Candlestik Done\n{END}'.format(**formatters))

# convert to candlestick chart testing data
print('{RED}\nConvert Testing Data to Candlestik{END}'.format(**formatters))
subprocess.call('python preprocess.py -m ohlc2cs -l {} -i stockdatas/{}_testing.csv -t testing'.format(
    windows_length, symbol), shell=True)
print('{GREEN}Convert Testing Data to Candlestik Done\n{END}'.format(**formatters))

# labelling data training
print('{RED}\nLabelling Training Data{END}'.format(**formatters))
subprocess.call('python preprocess.py -m img2dt -i dataset/{}/{}/training -lf {}_training_label_{}.txt'.format(
    windows_length, symbol, symbol, windows_length), shell=True)
print('{GREEN}Labelling Training Data Done\n{END}'.format(**formatters))

# labelling data testing
print('{RED}\nLabelling Testing Data{END}'.format(**formatters))
subprocess.call('python preprocess.py -m img2dt -i dataset/{}/{}/testing -lf {}_testing_label_{}.txt'.format(
    windows_length, symbol, symbol, windows_length), shell=True)
print('{GREEN}Labelling Testing Data Done\n{END}'.format(**formatters))

# print('{RED}Last step please resize images with your own.{END}'.format(**formatters))
# find . -maxdepth 4 -iname "*.png" | xargs -L1 -I{} convert -flatten +matte -adaptive-resize 200x200! "{}" "{}"

# find . -name "*.png" -exec convert "{}" -alpha off "{}" \;
os.system(
    'spd-say --voice-type female3 "your program has finished" --rate -50 --pitch 50')
