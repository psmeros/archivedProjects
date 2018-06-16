from .module import Module

class Optimizer(Module):
    """
    Abstract class for optimizers.
    """    
    def __init__(self, params, opt_params):
        '''Store locally model and optimizer parameter'''
        self.params = params
        self.opt_params = opt_params

    def zero_grad(self):
        for p in self.params:
            p['grad'].zero_()
        

class SGD(Optimizer):
    """
    Module for SGD optimizer.
    """    
    def __init__(self, params, lr=0.5, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        super(SGD, self).__init__(params, dict(lr=lr, weight_decay=weight_decay))

    def step(self):
        for p in self.params:
            #w = (1-lr*decay)*w - lr*w.grad            
            p['value'] = (1.0 - self.opt_params['lr']*self.opt_params['weight_decay']) * p['value'] - self.opt_params['lr'] * p['grad']
 
