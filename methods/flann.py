import open3d as o3d
import numpy as np
from methods.base_method import BaseMethod


class Flann(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)

  def prepare_input(self, x, y):
    tree = o3d.geometry.KDTreeFlann()
    tree.set_matrix_data(x.transpose())
    return tree, y

  def match(self, tree, query, k=None, radius=None):
    idx_list, dist_list = [], []
    for i in range(query.shape[0]):
      if self.search_method == 'knn':
        _, idx, dist = tree.search_knn_vector_xd(query[i, :], knn=k)
      elif self.search_method == 'radius':
        _, idx, dist = tree.search_radius_vector_xd(query[i, :], radius=radius)

      idx_list.append(idx)
      dist_list.append(dist)
    return np.array(dist_list), np.array(idx_list)
