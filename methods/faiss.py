from methods.base_method import BaseMethod
import faiss
import numpy as np

#https://github.com/facebookresearch/faiss/blob/master/tutorial/python/5-Multiple-GPUs.py


class Faiss(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)
    self.res = faiss.StandardGpuResources()

  def prepare_input(self, x, y, k):
    d = x.shape[-1]
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
    gpu_index.add(x)
    return gpu_index, y

  def match(self, tree, query, k):
    dist_list, idx_list = tree.search(query, k)
    return dist_list, idx_list