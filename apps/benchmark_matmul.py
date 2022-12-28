import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import needle as ndl
from needle import backend_ndarray as nd

from time import time

# doing matrix multiplication on a device and return the time that
# matrix multiplication takes in seconds.
def time_matmul(m, n, p, device):
    np.random.seed(1)
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    start = time()
    matmul_result = A @ B
    end = time()
    np.testing.assert_allclose(_A @ _B, matmul_result.numpy(), atol=1e-3, rtol=1e-3)
    return end - start

matrix_sizes = [10, 100, 200, 300, 400, 500, 900, 1000, 1500, 2000, 2500, 4000]
cpu_times = []
m1_times = []

for size in matrix_sizes:
    cpu_time = time_matmul(m = size, n = size, p = size, device = nd.cpu())
    m1_time = time_matmul(m = size, n = size, p = size, device = nd.m1())
    cpu_times.append(cpu_time)
    m1_times.append(m1_time)

speed_ups = [cpu_time / m1_time for cpu_time, m1_time in zip(cpu_times, m1_times)]

df = pd.DataFrame(list(zip(matrix_sizes, cpu_times, m1_times, speed_ups)), columns = ['matrix_size', 'cpu_time (s)', 'm1_time (s)', 'speedup'])
print(df)

df.plot(x = 'matrix_size', y = ['cpu_time (s)', 'm1_time (s)'], kind = 'bar', xlabel = 'matrix_size', logy = True, title = 'matmul duration vs matrix size (y axis is log scaled)')
plt.show()

plt.figure()
plt.scatter(matrix_sizes, speed_ups)
plt.title('matmul speedups comparing m1 to cpu')
plt.xlabel('matrix size (m = n = p = matrix size)')
plt.ylabel('speedups (cpu time / m1 time)')
plt.show()