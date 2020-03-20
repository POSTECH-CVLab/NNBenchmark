from methods.base_method import BaseMethod
from bufferkdtree.neighbors import NearestNeighbors

tree_depth = 9
plat_dev_ids = {0: [0]}


class BufferKD(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)

  def prepare_input(self, x, y, k):
    tree = NearestNeighbors(
        algorithm="buffer_kd_tree",
        plat_dev_ids=plat_dev_ids,
        tree_depth=9,
    )
    tree.fit(x)
    return tree, y

  def match(self, tree, query, k):
    dist_list, idx_list = tree.kneighbors(query, n_neighbors=k)
    return dist_list, idx_list
