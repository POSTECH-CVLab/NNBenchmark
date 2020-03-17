import numpy as np


def dataloader(num_iter, n, d, seed=1234):
  shape = (n, d)
  randg = np.random.RandomState(seed)
  for _ in range(num_iter):
    x = randg.rand(*shape).astype(np.float32)
    y = randg.rand(*shape).astype(np.float32)
    yield (x, y)
