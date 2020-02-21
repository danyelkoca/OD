# Source code for research: OD matrix estimation with deep learning using maps
# Author: Danyel Koca

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import r2_score
import numpy as np
import datetime

# Below are the graph convolution functions:
# (where each node collects information about nearby nodes)

def gcn_message(edges):
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Below is the pytorch module that defines the operations at each graph convolution layer

class gcnLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(gcn_layer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs #inputs: POI features
        g.send(g.edges(), gcn_message)  #send + receive implements the graph convolution function desribed by Kipf and Welling (2017)
        g.recv(g.nodes(), gcn_reduce)   # See https://tkipf.github.io/graph-convolutional-networks/  for the function and explanation.
        h = g.ndata.pop('h')    # Result(Convoluted POIs) of convolution at a layer is extracted
        return self.linear(h) # Result is linearly transformed

# Below is the pytorch class (machine learning architetures are initiliazed as classes) 
# that defines the the graph convolutional network (GCN) architecture (number of hidden layers, neurons, activation function, etc)

class gcn(torch.nn.Module):
    def __init__(self, input, hidden, output):
        super(gcn, self).__init__()
        # Initially each row in the input has (input) number of elements. 
        #In other words, each node in the network has (input number of features, i.e.: number of POI types)
        self.gcnInput = gcnLayer(input,hidden) # Input size is converted into hidden size
        self.gcnHidden = gcnLayer(hidden,hidden) # Hidden size is converted into hidden size 
        self.gcnOutput = gcnLayer(hidden,output) # Hidden size is converted into desired output size

    # Forward function: this function is run when we call the class
    def forward(self, g, pois):
        y = F.relu(self.gcnInput(g, pois)) # Result of the input layer is sent through activation function
        y = F.relu(self.gcnHidden(g, y)) # Result of the hidden layer is sent through activation function
        y = F.relu(self.gcnHidden(g, y)) # Result of the hidden layer is sent through activation function (Here, an arbitrary amount of hidden layers can be added)
        y = self.gcnOutput(g, y) # Result of the output layer (not activated)
        return y


# Below is the pytorch class that defines the the multilayer perceptron (MLP) architecture
# (number of hidden layers, neurons, activation function, etc)

class mlp(torch.nn.Module):
    def __init__(self, input, hidden):
        super(mlp, self).__init__() #initialize

        self.classifier = nn.Sequential( # Sequential is used when combining different layers
            nn.Linear(input, hidden), # Input feature matrix is converted into a matrix with shape (hidden) and linearly transformated
            nn.ReLU(), # Activation function is applied
            nn.Linear(hidden, hidden), # Result of previous layer is linearly transformaed
            nn.ReLU(), # Activation function is applied
            nn.Linear(hidden, 1))  # At the final layer, one output is given (Trip amount)

    def forward(self, x):
        x = self.classifier(x) # the input is sent throught the MLP architecture defined above
        return x

# Below is the pytorch class that defines the the the combined deep learning architecture

class od(nn.Module):
    def __init__(self, gcnInput, gcnHidden, gcnOutput, mlpHidden):
        super(od, self).__init__()
        self.gcn = gcn(gcnInput, gcnHidden,gcnOutput) # First: GCN
        self.mlp = mlp((2*gcnoutput+1), mlpHidden) # Afterwards: MLP


    def forward(self, g, pois, costs, indices, q, zoneCount):
        y = self.gcn(g,pois) # First, send the input through GCN
        p = torch.zeros(len(costs),2*q).cuda() # Prepare a matrix that will have the POI output at origin (size: q), POI output at destination (size: q) 
        count = 0 
        for i in range(zoneCount):
            for j in range(zoneCount):
                p[count][:q] = y[i][:] # POI output at origin (size: q)
                p[count][q:] = y[j][:] # POI output at destination (size: q) 
                count +=1
        p = p[indices][:] # Order the input matrix in the order of shuffled zones (or OD pairs)
        costs = costs[indices][:] # Order the cost matrix in the order of shuffled zones (or OD pairs)
        inputs = torch.cat((p, costs), 1).cuda() # Combine POI and cost matrices
        y = self.mlp(inputs) # Last, send through MLP
        return y

def train(optimizer, model, criterion, pois, costs, labels, indices, zoneCount, gcnOutput):
    model.train() # Model is in the training mode (meaning gradients are calculated)
    optimizer.zero_grad() # Gradients are zeroed
    pred = model(g, pois, costs, indices, gcnOutput, zoneCount) # Get model output as predicted output
    loss = criterion(pred, labels) # Calculate loss between prediction and label
    loss.backward() # Backpropagate the gradients
    optimizer.step() # (I dont fully know what happens with this code)
    return loss.item() # Return loss


def test(model, pois, costs, labels, indices, zoneCount, gcnOutput):
    model.eval() # Mode is in evaluation mode: no gradients are calcualted
    with torch.no_grad(): # In tensorflow if tensor has a parameter "autograd:true" then, gradients are calculated. This code sets the autograd to false for all tensors below
        pred = model(g, pois, costs, indices,gcnOutput, zoneCount) # Get prediction
        predictions = pred.detach().cpu() # Move prediction tensor from GPU to CPU
        r2 = r2_score(labels.cpu(), predictions) # Calculate R2
        return r2 

def data_collection(key): #Below part gets the data from the files into the program (POIS, nodes, costs, labels). If the file types are different than the ones used in this research, this part should be adjusted.
    if key == "mb": #mb: manhattan and brooklyn case
        no = 3
    else:
        no = 2
    with open("{}/nodes.csv".format(key)) as f:
        nodeCount = sum(1 for line in f)
    with open("{}/poisInfo.csv".format(key)) as f:
        poiCount = sum(1 for line in f)
    with open("{}/zones.csv".format(key)) as f:
        zoneCount = sum(1 for line in f)

    pois = torch.zeros((nodeCount,poiCount)).cuda()
    i = 0
    with open('{}/nodes.csv'.format(key), mode='r') as rx:
        r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in r:
            pois[i][:] = torch.FloatTensor([int(i) for i in row[no:]])
            i += 1

    costs = torch.zeros((zoneCount*zoneCount,1)).cuda()
    labels = torch.zeros((zoneCount*zoneCount,1)).cuda()
    count = 0
    with open('{}/costsTrips.csv'.format(key), mode='r') as rx:
        r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in r:
            costs[count][0] = int(row[2])
            labels[count][0] = int(row[3])
            count += 1

    g = dgl.DGLGraph().to(torch.device('cuda:0')) # dgl: deep graph learning library: We move POIs to the graph for graph convolution
    g.add_nodes(nodeCount) # Add nodes to the graph


    with open('{}/edges.csv'.format(key), mode='r') as rx:
        r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in r:
            g.add_edge(int(row[0]), int(row[1])) # If edge exists between 2 nodes, add edge

    print('We have %d nodes.' % g.number_of_nodes())
    print('We have %d edges.' % g.number_of_edges())


    return([g, pois, labels,costs, zoneCount, poiCount])

gcnoutput = 10
keys = ["manhattan", "brooklyn", "mb"]
count = 0
with open("costFinal.csv", mode='w', newline="") as wx:
    w = csv.writer(wx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w.writerow(["place", "iteration", "split", "r2"])
    for key in keys:
        [g, pois, labels, costs, zoneCount, poiCount] = data_collection(key)
        for iteration in range(1,11): # We test each split ratio with 10 times to get the average
            a = np.random.permutation(zoneCount) # randomize the zones
            for i in range(1,10):
                split = i/10 # Below lines split the training and test subsets
                breaker = int(split * zoneCount)
                train_zones = a[:breaker]
                test_zones = a[breaker:]
                train_indices = []
                test_indices = []
                for z in train_zones:
                    train_indices += [j for j in range(z * zoneCount, z * zoneCount + zoneCount)]
                for z in test_zones:
                    test_indices += [j for j in range(z * zoneCount, z * zoneCount + zoneCount)]
                # model parameters: gcninput, gcnhidden, gcnoutput, mlphidden
                model = od(poiCount, 64, gcnoutput, 64).cuda() # construct the model
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # optimizer: adam optimizer
                criterion = torch.nn.MSELoss() # loss: mean squared error loss
                for epoch in range(1, 501): # Train the algorithm 500 epochs
                    loss = train(optimizer, model, criterion, pois, costs, labels[train_indices], train_indices, zoneCount, gcnoutput) 
                    print(count, datetime.datetime.now() - start, key, iteration, i, epoch, loss)
                    count += 1
                r2 = test(model, pois, costs, labels[test_indices], test_indices, zoneCount, gcnoutput) # At the end of the algorithm, test the model and get r2
                w.writerow([key, iteration, i*10, r2]) # write key[manhattan,brooklyn,manhattan and brooklyn], iteration[0...9], split ratio[10%...90%], r2 to the file
