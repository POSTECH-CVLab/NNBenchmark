def load_method(method):
  if method == 'Flann':
    from methods.flann import Flann
    return Flann
  elif method == 'BufferKD':
    from methods.bufferkd import BufferKD
    return BufferKD
  elif method == 'Faiss':
    from methods.faiss import Faiss
    return Faiss
  elif method == 'KnnCUDA':
    from methods.knncuda import KnnCUDA
    return KnnCUDA
  else:
    raise ValueError('Method %s not found' % (method))
