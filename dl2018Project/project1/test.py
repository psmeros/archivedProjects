from _ctypes import resize

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
import pdb
import copy
# for timing
import timeit

##############################################################################
# base parameters
##############################################################################
max_epochs=50
batchsize=2
torch.manual_seed(42)
np.random.seed(42)
verbose = False
##############################################################################
# Load data/target of train/test
##############################################################################
import dlc_bci as bci


train_input , train_target = bci.load(root='./dataset/',one_khz = False)
test_input , test_target = bci.load(root = './dataset/', one_khz = False,train = False )
# k-fold cross validation
kf = KFold(n_splits=5, random_state=42)

train_dataset=torch.utils.data.TensorDataset(train_input, train_target)
test_dataset=torch.utils.data.TensorDataset(test_input, test_target)

# Base model
class Net(nn.Module):

    def __init__(self,nb_hidden, p_drop=0, nb_hidden2=1000):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2688, nb_hidden),
            nn.Linear(nb_hidden, 1)
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# Model without using MaxPool2D
class NetNoMaxPool2D(nn.Module):

    def __init__(self,nb_hidden, p_drop=0, nb_hidden2=1000):
        super(NetNoMaxPool2D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(44800, nb_hidden),
            nn.Linear(nb_hidden, 1)
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# Model without using BatchNorm
class NetNoBN(nn.Module):

    def __init__(self,nb_hidden, p_drop=0, nb_hidden2=1000):
        super(NetNoBN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Linear(2688, nb_hidden),
            nn.Linear(nb_hidden, 1)
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

#LinearClassifier classifier
class LinearClassifier(nn.Module):
    def __init__(self,):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(28*50,1)

    def forward(self, x):
        x = x.view(-1, 28*50)
        w_x = self.fc(x)
        return w_x


##############################################################################
# Define train functions
##############################################################################
# the training function, looping over each batch

from torch import FloatTensor as Tensor
def train(net,train_dataset,batchsize, l2=0.2, hinge=False):
    net.train()

    ##############################################################################
    # Choose Optimizer
    ##############################################################################
    optimizer = optim.SGD(net.parameters(), lr = 1e-3, weight_decay=l2)
    criterion = nn.BCEWithLogitsLoss()
    # If the loss is hingeloss
    if hinge:
        critertion = nn.HingeEmbeddingLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Iterate for each epoch
    ##############################################################################
    for epoch in range(max_epochs):
        sum_loss=0
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batchsize,
                                                   shuffle=True)
        for i, (images, labels) in enumerate(train_loader):
            images=torch.unsqueeze(images,1)
            labels=torch.unsqueeze(labels,1)
            labels=labels.float()

            data = Variable(images)
            target = Variable(labels)
            # If the loss is hinge loss
            if hinge:
                target[target==0] = -1

            ####### Optimize
            optimizer.zero_grad()
            ####### Calculate output
            output = net(data)
            ####### Calculate and update based on loss
            loss=criterion(output,target)
            # If the loss is hinge loss
            if hinge:
                #max(0, 1 - target*output)
                loss = torch.mean(torch.clamp(1 - target*output, min=0))
            else:
                loss = criterion(output,target)

            loss.backward()
            optimizer.step()

            ####### Sum loss
            sum_loss += loss.data.numpy()[0]
        if verbose:
            print ('Batchsize %d Tranining Epoch [%d/%d], Loss: %.4f' %(batchsize,epoch+1, max_epochs, sum_loss))


# the test function
def test(net,test_dataset,batchsize, hinge=False):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batchsize,
                                          shuffle=False)
    correct = 0
    total = 0
    for images, labels in test_loader:
        images=torch.unsqueeze(images,1)
        labels=torch.unsqueeze(labels,1)
        images = Variable(images, volatile=True)

        outputs = net(images)

        predicted = torch.round(torch.sigmoid(outputs.data))
        # If the loss is hinge loss
        if hinge:
            predicted = outputs.data
            predicted[predicted >=0] = 1
            predicted[predicted<0] = 0
        correct += accuracy_score(labels, predicted, normalize = False)
        total += labels.size(0)

    accuracy=100 * correct / total
    return accuracy

# Experiment by varying batchsize and nb_hidden
def get_best_batchsize_based_on_nbhidden(train_dataset,nb_hidden):
    batchsizeList=[2,4,8,16,32,64]
    accuracyList=[]
    for batchsize in batchsizeList:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=285,
                                                   shuffle=True)
        train_datasetRand=None
        for i, (images, labels) in enumerate(train_loader):
            if i==1:
                test_datasetRand=torch.utils.data.TensorDataset(images, labels)
                break
            train_datasetRand=torch.utils.data.TensorDataset(images, labels)
        net = Net(nb_hidden)
        train(net,train_datasetRand,batchsize)
        accuracy=test(net,test_datasetRand,batchsize)
        accuracyList.append(accuracy)

    index_max = np.argmax(accuracyList)
    #print('Nb_hidden %d found out the best batch size %d with accuracy %f'%(nb_hidden,batchsizeList[index_max],accuracyList[index_max]))
    return accuracyList[index_max],batchsizeList[index_max]

# The main experiment function
def experiment(net_o, batchsize, l2 = 0.2, hinge=False):
    accuracies = []
    for train_index, test_index in kf.split(train_input):
        net = copy.deepcopy(net_o)
        X_train, X_test = torch.index_select(train_input, 0, torch.LongTensor(train_index)), torch.index_select(train_input, 0, torch.LongTensor(test_index))
        y_train, y_test = torch.index_select(train_target, 0, torch.LongTensor(train_index)), torch.index_select(train_target, 0, torch.LongTensor(test_index))
        X_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        train(net, X_dataset, batchsize, l2, hinge)
        accuracy = test(net, val_dataset, batchsize, hinge)
        #print("validation accuracy: %f"%(accuracy))
        accuracies.append(accuracy)
    return sum(accuracies)/len(accuracies)

# Experiment by varying nb_hidden with different models
def experiment_with_nb_hidden(batchsize, bn=True, hinge=False, maxpool = True):
    nb_hidden_list=[ 2, 5, 10, 20, 50, 100, 150, 200]
    max_acc = 0
    best_nb_hidden = 0

    # Select best nb_hidden
    for nb_hidden in nb_hidden_list:
        start_time = timeit.default_timer()
        net = Net(nb_hidden, 0)
        if not bn:
            net = NetNoBN(nb_hidden, 0)
        if not maxpool:
            net = NetNoMaxPool2D(nb_hidden, 0)
        acc = experiment(net, batchsize, 0.2, hinge)
        print("#hidden: %f - accuracy: %f"%(nb_hidden, acc))
        if acc > max_acc:
            max_acc = acc
            best_nb_hidden = nb_hidden

    return best_nb_hidden

# Experiment by varying weight decay with different models
def experiment_with_weight_decay(batchsize, bn=True, hinge = False, maxpool = True):
    l2_list = [0, 0.1, 0.2, 0.3]
    max_acc = 0
    best_l2 = 0

    for l2 in l2_list:
        start_time = timeit.default_timer()
        net = Net(200, 0)
        if not bn:
            net = NetNoBN(200, 0)
        if not maxpool:
            net = NetNoMaxPool2D(200, 0)
        acc = experiment(net, batchsize, l2, hinge)
        print("l2: %f - accuracy: %f"%(l2, acc))
        if acc > max_acc:
            max_acc = acc
            best_l2 = l2
    return best_l2

# Experiment by varying dropout prob. with different models
def experiment_with_dropout(batchsize, bn=True, hinge = False, maxpool = True):
    nb_dropout_list=[ 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    max_acc = 0
    best_p_dropout = 0

    # Select best p_dropout
    for p_dropout in nb_dropout_list:
        start_time = timeit.default_timer()
        net = Net(200, p_dropout)
        if not bn:
            net = NetNoBN(200, p_dropout)
        if not maxpool:
            net = NetNoMaxPool2D(200, p_dropout)
        acc = experiment(net, batchsize, 0, hinge)
        print("#dropout probability: %f - accuracy: %f"%(p_dropout, acc))
        if acc > max_acc:
            max_acc = acc
            best_p_dropout = p_dropout

    return best_p_dropout

# Experiment with baselines from scikitlearn
def experiment_scikitlearn_baselines(train_X, train_Y, test_X, test_Y):
    train_X = train_X.numpy().reshape(316, 28*50)
    train_Y.numpy()

    sv = svm.SVC()
    sv.fit(train_X, train_Y)

    nn = NearestCentroid()
    nn.fit(train_X, train_Y)

    ga = GaussianNB()
    ga.fit(train_X, train_Y)

    dt = tree.DecisionTreeClassifier()
    dt.fit(train_X, train_Y)

    test_X = test_X.numpy().reshape(100, 28*50)
    test_Y.numpy()
    print("SVM " + str(accuracy_score(test_Y, sv.predict(test_X))))
    print("NN " + str(accuracy_score(test_Y, nn.predict(test_X))))
    print("Gausian " + str(accuracy_score(test_Y, ga.predict(test_X))))
    print("DT " + str(accuracy_score(test_Y, dt.predict(test_X))))
    print("Warning: The following is taking approximately 1.5 hours in an average laptop.")

# Experiment with nb_hidden and batchsize
def train_dnn():
    # Run n_hidden for Deep learning
    ##############################################################################
    nb_hidden_list=[2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400]
    accList=[]
    batchsizeList=[]

    for nb_hidden in nb_hidden_list:
        start_time = timeit.default_timer()
        acc,bestBatchsizeTemp=get_best_batchsize_based_on_nbhidden(train_dataset,nb_hidden)
        net = Net(nb_hidden)
        train(net,train_dataset,bestBatchsizeTemp)
        accuracy=test(net,test_dataset,bestBatchsizeTemp)
        elapsed = timeit.default_timer() - start_time
        print('Nb_hidden %d found out the best batch size %d with accuracy %f in duration time %f (s)'%(nb_hidden,bestBatchsizeTemp,
                                                                                accuracy,elapsed))
        accList.append(accuracy)
        batchsizeList.append(bestBatchsizeTemp)


    bestInd=np.argmax(accList)
    best_nb_hidden=nb_hidden_list[bestInd]
    bestAcc=accList[bestInd]
    bestBatchsize = batchsizeList[bestInd]

    print('The best accuracy is %f at nb_hidden %d and batch size %d'%(bestAcc,best_nb_hidden,bestBatchsize))

##############################################################################
# Main function
##############################################################################
if __name__ == '__main__':

    test_hingeloss = True
    test_bn = True
    test_wo_bn = True
    test_wo_maxpool =  True
    ################### Scikirlearn baselines ##################################
    experiment_scikitlearn_baselines(train_input, train_target, test_input, test_target)

    ################### Experiment with Batchnorm ###############################
    # Select best parameters using cross validation
    if test_bn:
        best_l2 = experiment_with_weight_decay(batchsize)
        best_nb_hidden = experiment_with_nb_hidden(batchsize)
        best_p_dropout = experiment_with_dropout(batchsize)


        # Use best parameters to train with whole training data
        net = Net(best_nb_hidden, best_p_dropout)
        train(net, train_dataset, batchsize, best_l2)
        # Use trained net with best parameters with test data
        print("Test acc w BatchNorm: %f"%(test(net, test_dataset, batchsize)))

    ################### Experiment without Batchnorm ############################
    #Select best parameters using cross validation
    if test_wo_bn:
        best_l2 = experiment_with_weight_decay(batchsize, False)
        best_nb_hidden = experiment_with_nb_hidden(batchsize, False)
        best_p_dropout = experiment_with_dropout(batchsize, False)

        net = NetNoBN(best_nb_hidden, best_p_dropout)
        train(net, train_dataset, batchsize, best_l2)
        print("Test acc w/o BatchNorm %f"%(test(net, test_dataset, batchsize)))

    ################### Experiment with HingeLoss ############################
    #Select best parameters using cross validation
    if test_hingeloss:
        best_l2 = experiment_with_weight_decay(batchsize, True, True)
        best_nb_hidden = experiment_with_nb_hidden(batchsize, True, True)
        best_p_dropout = experiment_with_dropout(batchsize, True, True)

        net = Net(best_nb_hidden, best_p_dropout)
        train(net, train_dataset, batchsize, best_l2)
        print("Test acc with HingeLoss: %f"%(test(net, test_dataset, batchsize)))


    ################### Experiment without Maxpool ###############################
    # Select best parameters using cross validation
    if test_wo_maxpool:
        best_l2 = experiment_with_weight_decay(batchsize, True, False, False)
        best_nb_hidden = experiment_with_nb_hidden(batchsize, True, False, False)
        best_p_dropout = experiment_with_dropout(batchsize, True, False, False)

        # Use best parameters to train with whole training data
        net = Net(best_nb_hidden, best_p_dropout)
        train(net, train_dataset, batchsize, best_l2)
        # Use trained net with best parameters with test data
        print("Test acc w/o MaxPool2D: %f"%(test(net, test_dataset, batchsize)))

    # Run LinearClassifier as a baseline
    ##############################################################################
    net = LinearClassifier()
    train(net,train_dataset,batchsize)
    accuracy=test(net,test_dataset,batchsize)
    print('Baseline LinearClassifier with accuracy :%f'%(accuracy))

    ################### Experiment of effect between batchsize and nb_hidden #####
    train_dnn()
