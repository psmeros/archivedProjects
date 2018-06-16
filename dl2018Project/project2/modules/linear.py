from torch import FloatTensor
from .module import Module

class LinearLayer(Module):
    """
    Abstract class for linear layers.
    """

class Linear(LinearLayer):
    """
    Module for linear transformation.
    """
    def __init__(self, inputSize, outputSize):
        #Initialize weights and biases with 0
        w = FloatTensor(outputSize, inputSize).zero_()
        b = FloatTensor(outputSize).zero_()
        dw = FloatTensor(outputSize, inputSize).zero_()
        db = FloatTensor(outputSize).zero_()
        self.params = [{'value':w, 'grad':dw}, {'value':b, 'grad':db}]

    def forward(self, input):
        self.input = input
        #y = x*wT + b
        out = input.matmul(self.params[0]['value'].t()) + self.params[1]['value']
        return out

    def backward(self, gradwrtoutput):
        #dw = gradwrtoutput * (x*wT +b)' = gradwrtoutput * x
        self.params[0]['grad'] = gradwrtoutput.t().matmul(self.input)
        #db = gradwrtoutput * (x*wT +b)' = gradwrtoutput * 1
        self.params[1]['grad'] = gradwrtoutput.sum(dim=0)
        #dx = gradwrtoutput * (x*wT +b)' = gradwrtoutput * w
        dx = gradwrtoutput.matmul(self.params[0]['value'])
        return dx

    def param(self):
        return self.params