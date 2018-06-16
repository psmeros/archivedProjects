import numpy as np
import random
import math
import pdb
from torch import FloatTensor
from modules.module import Module
from modules.container import Sequential
from modules.linear import Linear
from modules.activation import ReLU, Tanh
from modules.loss import MSELoss, MAELoss
from modules.optimizer import SGD
from sanity_check import check


#generate random data
def generate_data():
    random.seed(1)
    np.random.seed(1)

    samples = np.random.uniform(0,1,(1000,2))

    def distance(sample):
        return math.sqrt((sample[0] - 0.5)*(sample[0] - 0.5)+(sample[1]-0.5)*(sample[1]-0.5))

    labels = []
    for sample in samples:
        if distance(sample) < 1/(math.sqrt(2*math.pi)):
            labels.append(1)
        else:
            labels.append(0)
    return samples, labels

if __name__ == "__main__":
    #Compare in terms of results our implementation with PyTorch
    sanity_check = False
    if sanity_check:
        check()

    # Hyper Parameters
    num_epochs = 60
    learning_rate = 1.e-5
    weight_decay = 0.0


    # Network with two input units, two output units and three hidden layers of 25 units
    model = Sequential(
        Linear(2, 25),
        ReLU(),
        Linear(25, 2)
        )

    # Data generation
    data, labels = generate_data()

    # Model training        
    optimizer = SGD(model.param(), lr=learning_rate, weight_decay=weight_decay) 
 
    for epoch in range(num_epochs):
        inputs = FloatTensor(data)
        targets = FloatTensor(labels).view(len(labels), -1)

        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = MSELoss(outputs, targets)
        loss = criterion()
        model.backward(criterion.backward())
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print ('Epoch [%d/%d], Loss: %.4f' 
                %(epoch+1, num_epochs, loss))
