from methods.base_method import BaseMethod
import knn

class KnnCUDA(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)

  def prepare_input(self, x, y):
    return x, y

  def match(self, tree, query):
    x = tree
    dist_list, idx_list = knn.knn(x.reshape(self.opt.dimenstion, -1), y.reshape(self.opt.dimenstion, -1), self.knn)
