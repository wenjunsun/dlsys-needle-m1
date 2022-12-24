# Download the datasets you will be using for this assignment

import urllib.request
import os
import tarfile

# !mkdir -p './data/ptb'
os.makedirs('./data/ptb', exist_ok = True)

# Download Penn Treebank dataset
ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for f in ['train.txt', 'test.txt', 'valid.txt']:
    if not os.path.exists(os.path.join('./data/ptb', f)):
        urllib.request.urlretrieve(ptb_data + f, os.path.join('./data/ptb', f))

# Download CIFAR-10 dataset
if not os.path.isdir("./data/cifar-10-batches-py"):
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "./data/cifar-10-python.tar.gz")
    # !tar -xvzf './data/cifar-10-python.tar.gz' -C './data'
    file = tarfile.open('./data/cifar-10-python.tar.gz')
    file.extractall('./data')
    file.close()
    