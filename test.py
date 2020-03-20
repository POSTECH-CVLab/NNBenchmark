import argparse
import errno
import os
import sys
import time

import numpy as np
import pandas as pd

from dataloader import dataloader
from methods import load_method
from timer import Timer


def match(method, n, d, k, num_iter):
  dataset = dataloader(num_iter=num_iter, n=n, d=d)
  prepare_timer, match_timer, total_timer = Timer(), Timer(), Timer()
  min_k = min(n, k)

  i = 1
  for x, y in dataset:
    total_timer.tic()
    prepare_timer.tic()
    tree, query = method.prepare_input(x, y, min_k)
    prepare_timer.toc()

    match_timer.tic()
    result = method.match(tree, query, min_k)
    match_timer.toc()

    total_timer.toc()

    sys.stdout.write(
        "\r [%d/%d] N %d, D %d, K %d, prepare_time %.4f, matching time %.4f, total_time %.4f"
        % (i, num_iter, n, d, k, prepare_timer.avg, match_timer.avg, total_timer.avg))
    sys.stdout.flush()
    i += 1
  print("")
  return [n, d, k, prepare_timer.avg, match_timer.avg, total_timer.avg]


def sanity_check(method):
  n, d, k = 100, 16, 1
  dataset = dataloader(num_iter=5, n=n, d=d)

  is_pass = True
  for x, _ in dataset:
    tree, query = method.prepare_input(x, x, k)
    dists, inds = method.match(tree, query, k)
    if np.mean(dists[:, 0]) > 1e-5 or not np.array_equal(inds[:, 0], np.arange(0, n)):
      is_pass = False

  print(is_pass)


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  #Python > 2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_iter', type=int, default=20, help='number of iteration per scale')
  parser.add_argument('--max_scale', type=int, default=10, help='log_10(max scale)')
  parser.add_argument('--max_dim', type=int, default=10, help='log_2(max dim')
  parser.add_argument('--max_knn', type=int, default=10, help='log_2(max knn)')
  parser.add_argument('--method', type=str, default='Flann', help='name of method')
  parser.add_argument(
      '--search_method', type=str, default='knn', choices=['knn', 'radius'])
  parser.add_argument('--radius', type=float, default=0.1)
  parser.add_argument('--out_dir', type=str, default='./outputs')
  parser.add_argument('--sanity', action='store_true')

  opt = parser.parse_args()

  Method = load_method(opt.method)
  method = Method(opt)

  if opt.sanity:
    sanity_check(method)
  else:
    n_list = [pow(10, s) for s in range(1, opt.max_scale + 1)]
    d_list = [pow(2, d) for d in range(0, opt.max_dim + 1)]
    k_list = [pow(2, k) for k in range(0, opt.max_knn + 1)]

    results = []
    sweep = [[pow(10,5), d, 1] for d in d_list]
    sweep.extend([[pow(10,5), 4, k] for k in k_list])
    sweep.extend([[n, 3, 1] for n in n_list])
    
    mkdir_p(opt.out_dir)
    timestr = time.strftime("_%Y%m%d-%H%M%S")
    filename = os.path.join(opt.out_dir, opt.method + timestr)

    for n, d, k in sweep:
      with open(filename, 'ab') as f:
        result = match(method, n, d, k, opt.num_iter)
        np.savetxt(f, [result], fmt="%d %d %d %.4f %.4f %.4f")
      results.append(result)
