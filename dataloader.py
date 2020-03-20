import numpy as np


def dataloader(num_iter, n, d, seed=1234):
  randg = np.random.RandomState(seed)
  for _ in range(num_iter):
    x = randg.rand(n, d).astype(np.float32)
    y = randg.rand(10, d).astype(np.float32)
    yield (x, y)
