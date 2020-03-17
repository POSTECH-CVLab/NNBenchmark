from methods.base_method import BaseMethod
import faiss
import numpy as np

#https://github.com/facebookresearch/faiss/blob/master/tutorial/python/5-Multiple-GPUs.py

class Faiss(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)

  def prepare_input(self, x, y):
    x = x.astype('float32')
    cpu_index = faiss.IndexFlatL2(self.opt.dimension)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(x)
    return gpu_index, y

  def match(self, tree, query):
    query = query.astype('float32')
    dist_list, index_list = tree.search(query, self.knn)