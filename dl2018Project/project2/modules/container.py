from collections import OrderedDict, Iterable
from .module import Module

class Container(Module):
    """
    Abstract class for containers.
    """

class Sequential(Container):
    """
    Sequential container.
    Usage example:
        model = Sequential(
                  Linear(2, 25),
                  ReLU(),
                  Linear(25, 2)
                  )
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        # Dict to save module. Reason is to take advantage of order in OrderedDict
        self._modules = OrderedDict()
        # Allocate module into Dict
        for idx, module in enumerate(args):
            self._modules[str(idx)]=module

    def forward(self, input):
        # Call forward function of each  module for the input value
        for module in self._modules.values():
            input = module.forward(input)
        return input

    def backward(self, gradwrtoutput):
        # Inverse the list of module
        inversedModuleList=reversed(self._modules.values())
        # Call backward function of each  module for the input value
        for module in inversedModuleList:
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

    def param(self):
        # Create a list to contain parameter
        paramList=list()
        # Form a list containing param of all modules
        for module in self._modules.values():
            for param in module.param():
                paramList.append(param)
        return paramList

