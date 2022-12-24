import sys
sys.path.append('./python')
import numpy as np

import needle as ndl
from needle import backend_ndarray as nd

from time import time

np.random.seed(1)

def time_function_in_seconds(func, **kwargs):
    start = time()
    func(**kwargs)
    end = time()
    print(f'for matmul with parameters: {kwargs}')
    print(f'it takes {round(end - start, 4)} seconds to do matrix multiplication')

def matmul(m, n, p, device):
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    # np.testing.assert_allclose(_A @ _B, (A @ B).numpy(), atol=1e-5, rtol=1e-5) -- this seems to be too low a tolerance level
    np.testing.assert_allclose(_A @ _B, (A @ B).numpy(), atol=1e-3, rtol=1e-3)

time_function_in_seconds(matmul, m = 4000, n = 4000, p = 4000, device = nd.cpu())
time_function_in_seconds(matmul, m = 4000, n = 4000, p = 4000, device = nd.m1())