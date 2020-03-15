from methods.base_method import BaseMethod


class KnnCUDA(BaseMethod):

  def __init__(self, opt):
    BaseMethod.__init__(self, opt)

  def prepare_input(self, x, y):
    pass

  def match(self, tree, query):
    pass
