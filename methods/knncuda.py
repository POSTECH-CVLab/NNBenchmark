from methods.base_method import BaseMethod
import sys
sys.path.append("/root/code/knn_cuda/knn_cuda")

import knn


class KnnCUDA(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)
    if self.search_method == 'radius':
      raise ValueError(f'search method radius is not supported for {self.__class__.__name__} ')

  def prepare_input(self, x, y):
    return x.T, y.T

  def match(self, tree, query, k=None, radius=None):
    dist_list, idx_list = knn.knn(tree, query, k)
    dist_list = dist_list.T
    
    # index basis is 1
    idx_list = idx_list.T - 1
    return dist_list, idx_list
