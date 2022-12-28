import itertools
import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd


np.random.seed(2)


_DEVICES = [nd.cpu(), 
            pytest.param(nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")),
            pytest.param(nd.m1(), marks=pytest.mark.skipif(not nd.m1().enabled(), reason="No M1 Chip"))]


TRAIN = [True, False]
@pytest.mark.parametrize("train", TRAIN)
def test_cifar10_dataset(train):
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
    if train:
        assert len(dataset) == 50000
    else:
        assert len(dataset) == 10000
    example = dataset[np.random.randint(len(dataset))]
    assert(isinstance(example, tuple))
    X, y = example
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 32, 32)


BATCH_SIZES = [1, 15]
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda", "m1"])
def test_cifar10_loader(batch_size, train, device):
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, batch_size)
    for (X, y) in train_loader:
        break
    assert isinstance(X.cached_data, nd.NDArray)
    assert isinstance(X, ndl.Tensor)
    assert isinstance(y, ndl.Tensor)
    assert X.dtype == 'float32'


BPTT = [3, 32]
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("bptt", BPTT)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda", "m1"])
def test_ptb_dataset(batch_size, bptt, train, device):
    # TODO update with more tests?
    corpus = ndl.data.Corpus("data/ptb")
    if train:
        data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    else:
        data = ndl.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
    X, y = ndl.data.get_batch(data, np.random.randint(len(data)), bptt, device=device)
    assert X.shape == (bptt, batch_size)
    assert y.shape == (bptt * batch_size,)
    assert isinstance(X, ndl.Tensor)
    assert X.dtype == 'float32'
    assert X.device == device
    assert isinstance(X.cached_data, nd.NDArray)
    ntokens = len(corpus.dictionary)
    assert ntokens == 10000
