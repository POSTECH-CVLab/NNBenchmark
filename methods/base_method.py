class BaseMethod():

  def __init__(self, opt):
    self.opt = opt
    self.search_method = opt.search_method
    

  def prepare_input(self, x, y):
    pass

  def match(self, tree, query, k=None, radius=None):
    pass