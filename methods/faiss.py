from methods.base_method import BaseMethod
import faiss
import numpy as np

#https://github.com/facebookresearch/faiss/blob/master/tutorial/python/5-Multiple-GPUs.py


class Faiss(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)

  def prepare_input(self, x, y):
    cpu_index = faiss.IndexFlatL2(self.opt.dimension)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(x)
    return gpu_index, y

  def match(self, tree, query):
    dist_list, idx_list = tree.search(query, self.knn)
    return dist_list, idx_list