from methods.flann import Flann
# TODO: import methods

# TODO: append methods
METHODS = [Flann]

methods_str_mapping = {m.__name__: m for m in METHODS}


def load_method(method):
  if method in methods_str_mapping.keys():
    return methods_str_mapping[method]
  else:
    raise ValueError(f'Method {method} not found')
