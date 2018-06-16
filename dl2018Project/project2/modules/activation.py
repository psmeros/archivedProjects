from .module import Module

class ActivationFunction(Module):
    """
    Abstract class for activation functions.
    """

class ReLU(ActivationFunction):
    """
    Module for ReLU activation function.
    """    
    def forward(self, input):
        self.input = input
        mask = input > 0
        out = input*mask.float()	
        return out
        
    def backward(self, gradwrtoutput):
        mask = self.input > 0
        dx = gradwrtoutput*mask.float()
        return dx

    def param(self):
        return []

class Tanh(ActivationFunction):
    """
    Module for Tanh activation function.
    """        
    def forward(self, input):
        out = input.tanh()
        self.input = input
        return out
	
    # https://socratic.org/questions/what-is-the-derivative-of-tanh-x
    def backward(self, gradwrtoutput):
        grad = 1-self.input.tanh()**2
        dx = grad*gradwrtoutput
        return dx

    def param(self):
        return []
