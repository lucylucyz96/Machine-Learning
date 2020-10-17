""" Model classes defined here! """

import torch
import torch.nn.functional as F

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        # TODO: Implement this!
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

        

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 10, kernel_size=n2_kern, stride=2)
        

    def forward(self, x):
        #print(x.size())
        sample_size = x.size()[0]
        x = x.view(sample_size, 1, 28,28)

        # TODO: Implement this!
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #self.pool = torch.nn.MaxPool2d(x.size()[2],x.size()[3])
        self.pool = torch.nn.MaxPool2d(5)
        x = self.pool(x)
        #print(x.size())
        return x.view(sample_size,-1)

        raise NotImplementedError()

class BestNN(torch.nn.Module):
    # TODO: You can change the parameters to the init method if you need to
    # take hyperparameters from the command line args!
    def __init__(self):
        super(BestNN,self).__init__()
        # self.conv1 = torch.nn.Conv2d(1, 80, kernel_size=10)
        # self.batchnorm1 = torch.nn.BatchNorm2d(80)
        # self.conv2 = torch.nn.Conv2d(80,40,kernel_size = 5)
        # #self.conv3 = torch.nn.Conv2d(60,10,kernel_size = 5)
        # self.batchnorm2 = torch.nn.BatchNorm2d(40)
        # self.maxpool = torch.nn.MaxPool2d(2)
        # #self.dropout = torch.nn.Dropout()
        # self.linear1 = torch.nn.Linear(160, 20)
        # self.linear2 = torch.nn.Linear(20, 10)
        # #raise NotImplementedError()

        self.conv1 = torch.nn.Conv2d(1, 40, kernel_size=9, padding=4)
        self.batchnorm1 = torch.nn.BatchNorm2d(40)
        #self.conv2 = torch.nn.Conv2d(20,40,kernel_size = 5, padding=2)
        self.conv3 = torch.nn.Conv2d(40,80,kernel_size = 5, padding=2)

        self.batchnorm2 = torch.nn.BatchNorm2d(40)
        self.batchnorm3= torch.nn.BatchNorm2d(80)
        self.maxpool = torch.nn.MaxPool2d(2)
        #self.dropout = torch.nn.Dropout()
        #self.linear1 = torch.nn.Linear(15680, 720)
        self.linear1 = torch.nn.Linear(720, 120)#original :120
        self.linear2 = torch.nn.Linear(120, 10)
        #raise NotImplementedError()

    def forward(self, x):
        sample_size = x.size()[0]
        x = x.view(sample_size, 1, 28,28)

        x= self.conv1(x)
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.maxpool(x)

        #x = self.conv2(x)
        #x = self.batchnorm2(x)
        #x = F.relu(x)
        #x = self.dropout(x)
        #x = self.maxpool(x)

        x= self.conv3(x)
        x = self.batchnorm3(x)
        #x = self.maxpool(x)
        x = F.relu(x)
        x = self.maxpool(x)

        #x = F.relu(x)
        #print(x.size())
        x = x.view(x.size()[0],-1)
        x = F.relu(self.linear1(x))
        x = (self.linear2(x))
        #x = (self.linear3(x))
        return x
        #x = torch.nn.Linear(28*28)
        #raise NotImplementedError()
