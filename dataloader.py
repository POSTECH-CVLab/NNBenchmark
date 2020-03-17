import numpy as np


def dataloader(shape, num_iter, seed=1234):
  randg = np.random.RandomState(seed)
  for _ in range(num_iter):
    x = randg.rand(*shape).astype(np.float32)
    y = randg.rand(*shape).astype(np.float32)
    yield (x, y)
