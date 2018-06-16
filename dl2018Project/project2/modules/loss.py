from .module import Module

class Loss(Module):
    """
    Abstract class for losses.
    """    
    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets

class MSELoss(Loss):
    """
    Module for MSE Loss
    """
    def __init__(self, outputs, targets):
        super(MSELoss, self).__init__(outputs, targets)

    def forward(self):
        # mean((y' - y)^2)
        return ((self.targets - self.outputs) ** 2).mean()

    def backward(self):
        # -2(y' - y)
        return -2*(self.targets - self.outputs)
        
class MAELoss(Loss):
    """
    Module for MAE Loss
    """
    def __init__(self, outputs, targets):
        super(MAELoss, self).__init__(outputs, targets)

    def forward(self):
        # mean(|y' - y|)
        return (self.targets - self.outputs).abs().mean()

    def backward(self):
        # -1 if y' > y, 1 if y' < y, 0 if y' = y (non-differentiable; we pick 0)
        mask1 = self.targets > self.outputs
        mask2 = self.targets < self.outputs
        return (-1 * mask1.float() + mask2.float())
