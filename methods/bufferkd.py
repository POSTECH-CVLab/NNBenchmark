from methods.base_method import BaseMethod
from bufferkdtree.neighbors import NearestNeighbors

tree_depth = 9
plat_dev_ids = {0:[0]}

class BufferKD(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)

  def prepare_input(self, x, y):
    if self.search_method == 'knn':
      # if datasets are too small, bufferkdtree can't deal with it
      tree = NearestNeighbors(n_neighbors = self.knn, algorithm="buffer_kd_tree", plat_dev_ids=plat_dev_ids, tree_depth=9)
      tree.fit(x)
      return tree, y
    elif self.search_method == 'radius':
      raise ValueError('bufferkdtree has no search method : %s' % (self.search_method))

  def match(self, tree, query):
    dist_list, idx_list = tree.kneighbors(query)
