import argparse
import itertools
import os
import sys

import pandas as pd

from dataloader import dataloader
from methods import load_method
from timer import Timer


def match(method, n, d, num_iter):
  dataset = dataloader(num_iter=num_iter, n=n, d=d)
  prepare_timer, match_timer, total_timer = Timer(), Timer(), Timer()

  i = 1
  for x, y in dataset:
    total_timer.tic()
    prepare_timer.tic()
    tree, query = method.prepare_input(x, y)
    prepare_timer.toc()

    match_timer.tic()
    result = method.match(tree, query)
    match_timer.toc()

    sys.stdout.write("\r [%d/%d] " % (i, num_iter))
    sys.stdout.flush()
    i += 1
    total_timer.toc()

  return (n, d, prepare_timer.avg, match_timer.avg, total_timer.avg)


def sanity_check(method):
  n, d = 100, 16
  dataset = dataloader(num_iter=5, n=n, d=d)

  is_pass = True
  for x, _ in dataset:
    tree, query = method.prepare_input(x, x)
    dists, inds = method.match(tree, query)
    if np.sum(dists[:, 0]) > 0 or np.array_equal(inds[:, 0], np.arange(0, n)):
      is_pass = False

  print(is_pass)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_iter', type=int, default=20, help='number of iteration per scale')
  parser.add_argument('--max_scale', type=int, default=9, help='number of scales')
  parser.add_argument('--max_dim', type=int, default=1, help='size of dimension')
  parser.add_argument('--method', type=str, default='Flann', help='name of method')
  parser.add_argument(
      '--search_method', type=str, default='knn', choices=['knn', 'radius'])
  parser.add_argument('--knn', type=int, default=1)
  parser.add_argument('--radius', type=float, default=0.1)
  parser.add_argument('--out_dir', type=str, default='.')
  parser.add_argument('--sanity', action='store_true')

  opt = parser.parse_args()

  Method = load_method(opt.method)
  method = Method(opt)

  if opt.sanity:
    sanity_check(method)
  else:
    n_list = [pow(10, s) for s in range(1, opt.max_scale + 1)]
    d_list = [pow(2, d) for d in range(0, opt.max_dim)]

    sweep = list(itertools.product(n_list, d_list))

    results = []
    for n, d in sweep:
      result = match(method, n, d, opt.num_iter)
      print("N %d, D %d, prepare time %.4f, matching time %.4f, total time %.4f" %
            result)
      results.append(result)

    filename = os.path.join(opt.out_dir, opt.method)
    df = pd.DataFrame(results, columns=['n', 'd', 'prepare', 'match', 'total'])
    df.to_csv(filename)