import argparse
import os

from dataloader import dataloader
from methods import load_method
from timer import Timer


def match(method, num_iter, shape):
  d = dataloader(shape=shape, num_iter=num_iter)
  prepare_timer, match_timer = Timer(), Timer()

  for x, y in d:
    prepare_timer.tic()
    tree, query = method.prepare_input(x, y)
    prepare_timer.toc()

    match_timer.tic()
    method.match(tree, query)
    match_timer.toc()

    print("prepare time: %.4f, match time: %.4f" % (prepare_timer.avg, match_timer.avg))

  return prepare_timer.avg, match_timer.avg


def sanity_check(method):
  pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_iter', type=int, default=20, help='number of iteration per scale')
  parser.add_argument('--num_scale', type=int, default=9, help='number of scales')
  parser.add_argument('--dimension', type=int, default=1, help='size of dimension')
  parser.add_argument('--method', type=str, default='Flann', help='name of method')
  parser.add_argument(
      '--search_method', type=str, default='knn', choices=['knn', 'radius'])
  parser.add_argument('--knn', type=int, default=1)
  parser.add_argument('--radius', type=float, default=0.1)

  opt = parser.parse_args()

  method = load_method(opt.method)
  scales = [pow(10, s) for s in range(0, opt.num_scale)]

  for scale in scales:
    shape = (scale, opt.dimension)
    result = match(method, opt.num_iter, shape)
    print(result)