import numpy as np

class Array():
    """Array object for multi-purpose
    Array(self, array)
        - Array().numpy : return np.array()
    
    Method:
        - Array().torch() : convert array as torch.Tensor
    """
    def __init__(self, array, copy=False):
        if isinstance(array, np.ndarray) and not copy:
            self.numpy = array
        else:
            self.numpy = np.array(array)
    
    def __repr__(self):
        return f'<obj.{self.__class__.__name__} {self.shape}>'
    
    def torch(self, device=None):
        """
        Keyword Arguments:
            device {str} -- None or "cpu", "cuda", "cuda:0", ...
        """
        import torch
        _array = torch.from_numpy(self.numpy)
        if device:
            _device = torch.device(device)
            _array = _array.to(_device)
        
        return _array
    
    @property
    def shape(self):
        return self.numpy.shape