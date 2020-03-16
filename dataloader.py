import numpy as np


def dataloader(shape, num_iter, seed=1234):
  randg = np.random.RandomState(seed)
  for _ in range(num_iter):
    x = randg.rand(*shape)
    y = randg.rand(*shape)
    yield (x, y)
