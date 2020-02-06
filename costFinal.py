import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import r2_score
import numpy as np
import datetime
start = datetime.datetime.now()


def gcn_message(edges):
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

class gcn_layer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(gcn_layer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        h = g.ndata.pop('h')
        return self.linear(h)

class gcn(torch.nn.Module):
    def __init__(self, input, hidden, output):
        super(gcn, self).__init__()
        self.gcn_first = gcn_layer(input,hidden)
        self.gcn_hidden = gcn_layer(hidden,hidden)
        self.gcn_last = gcn_layer(hidden,output)


    def forward(self, g, pois):
        y = F.relu(self.gcn_first(g, pois))
        y = F.relu(self.gcn_hidden(g, y))
        #y = F.relu(self.gcn_hidden(g, y))
        #y = F.relu(self.gcn_hidden(g, y))
        #y = F.relu(self.gcn_hidden(g, y))
        #y = F.relu(self.gcn_hidden(g, y))
        y = F.relu(self.gcn_hidden(g, y))
        y = self.gcn_last(g, y)
        return y

class mlp(torch.nn.Module):
    def __init__(self, input, hidden):
        super(mlp, self).__init__()


        self.classifier = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = self.classifier(x)
        return x

class od(nn.Module):
    def __init__(self, gcninput, gcnhidden, gcnOutput, mlphidden):
        super(od, self).__init__()
        self.gcn = gcn(gcninput, gcnhidden,gcnOutput)
        self.mlp = mlp((2*gcnoutput+1), mlphidden)


    def forward(self, g, pois,costs,indices,gzn, zoneCount):
        y = self.gcn(g,pois)
        p = torch.zeros(len(costs),2*gzn).cuda()
        count = 0
        for i in range(zoneCount):
            for j in range(zoneCount):
                p[count][:gzn] = y[i][:]
                p[count][gzn:] = y[j][:]
                count +=1
        p = p[indices][:]
        costs = costs[indices][:]
        inputs = torch.cat((p, costs), 1).cuda()
        y = self.mlp(inputs)
        return y

def train(optimizer, model, criterion, pois, costs,labels,indices, zoneCount, gcnOutput):
    model.train()
    optimizer.zero_grad()
    pred = model(g,pois,costs,indices,gcnOutput, zoneCount)
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, pois, costs, labels, indices, zoneCount, gcnOutput):
    model.eval()
    with torch.no_grad():
        pred = model(g, pois, costs, indices,gcnOutput, zoneCount)
        predictions = pred.detach().cpu()
        r2 = r2_score(labels.cpu(), predictions)
        return r2

def data_collection(key):
    if key == "mb":
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

    g = dgl.DGLGraph().to(torch.device('cuda:0'))
    g.add_nodes(nodeCount)


    with open('{}/edges.csv'.format(key), mode='r') as rx:
        r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in r:
            g.add_edge(int(row[0]), int(row[1]))

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
        for iteration in range(1,11):
            a = np.random.permutation(zoneCount)
            for i in range(1,10):
                split = i/10
                breaker = int(split * zoneCount)
                train_zones = a[:breaker]
                test_zones = a[breaker:]
                train_indices = []
                test_indices = []
                for z in train_zones:
                    train_indices += [j for j in range(z * zoneCount, z * zoneCount + zoneCount)]
                for z in test_zones:
                    test_indices += [j for j in range(z * zoneCount, z * zoneCount + zoneCount)]
                # (gcninput, gcnhidden, gcnoutput, mlphidden):
                model = od(poiCount, 64, gcnoutput, 64).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                criterion = torch.nn.MSELoss()
                for epoch in range(1, 501):
                    loss = train(optimizer, model, criterion, pois, costs, labels[train_indices], train_indices, zoneCount, gcnoutput)
                    print(count, datetime.datetime.now() - start, key, iteration, i, epoch, loss)
                    count += 1
                r2 = test(model, pois, costs, labels[test_indices], test_indices, zoneCount, gcnoutput)
                w.writerow([key, iteration, i*10, r2])
"""
!!! PROGRAM RUNNING !!!
!!! PLEASE DONT STOP IT !!!
"""