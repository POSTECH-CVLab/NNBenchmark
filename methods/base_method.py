#from abc import ABC, abstractmethod

#class BaseMethod(ABC):
class BaseMethod():
  def __init__(self, opt):
    self.opt = opt
    self.search_method = opt.search_method
    self.knn = opt.knn
    self.radius = opt.radius

  #@abstractmethod
  def prepare_input(self, x, y):
    pass

  #@abstractmethod
  def match(self, tree, query):
    pass