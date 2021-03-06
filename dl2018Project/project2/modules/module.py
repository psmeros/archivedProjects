class Module(object):
    """
    Abstract class for Module.
    """
    def forward(self, *input):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
    def __call__(self, *input):
        return self.forward(*input)