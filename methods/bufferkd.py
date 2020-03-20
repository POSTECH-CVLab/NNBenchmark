from methods.base_method import BaseMethod
from bufferkdtree.neighbors import NearestNeighbors

tree_depth = 9
plat_dev_ids = {0: [0]}


class BufferKD(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)
    if self.search_method == 'radius':
      raise ValueError(f'search method radius is not supported for {self.__class__.__name__} ')

  def prepare_input(self, x, y):
    tree = NearestNeighbors(
        algorithm="buffer_kd_tree",
        plat_dev_ids=plat_dev_ids,
        tree_depth=9,
    )
    tree.fit(x)
    return tree, y

  def match(self, tree, query, k=None, radius=None):
    dist_list, idx_list = tree.kneighbors(query, n_neighbors=k)
    return dist_list, idx_list
