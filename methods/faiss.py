from methods.base_method import BaseMethod
import faiss
import numpy as np

#https://github.com/facebookresearch/faiss/blob/master/tutorial/python/5-Multiple-GPUs.py


class Faiss(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)
    self.faiss_index_string = self.config.faiss_index_string
    assert self.faiss_index_string, "require index string"

    self.res = faiss.StandardGpuResources()
    self.dev_no = 0

  def prepare_input(self, x, y):
    d = x.shape[-1]
    if self.faiss_index_string == 'Flat':
      index = faiss.IndexFlatL2(d)
    else:
      index = faiss.index_factory(d, self.faiss_index_string)
    index = faiss.index_cpu_to_gpu(self.res, self.dev_no, index)
    index.train(x)
    index.add(x)
    return index, y

  def match(self, tree, query, k=None, radius=None):
    if self.search_method == 'knn':
      dist_list, idx_list = tree.search(query, k)
    elif self.search_method == 'radius':
      dist_list, idx_list = tree.range_search(query, radius)
    return dist_list, idx_list