import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import r2_score
import scipy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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
        p = torch.zeros(len(costs),2*gzn)
        count = 0
        for i in range(zoneCount):
            for j in range(zoneCount):
                p[count][:gzn] = y[i][:]
                p[count][gzn:] = y[j][:]
                count +=1
        p = p[indices][:]
        costs = costs[indices][:]
        inputs = torch.cat((p, costs), 1)
        y = self.mlp(inputs)
        return y


def train(optimizer, model, criterion, pois, costs,labels,indices, zoneCount, gcnOutput):
    model.train()
    optimizer.zero_grad()
    # CHANGE
    pred = model(g,pois,costs,indices,gcnOutput, zoneCount)
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()


    return loss.item()


def test(model, criterion, pois, costs, labels, indices, epoch, zoneCount, gcnOutput, key, split):
    model.eval()
    with torch.no_grad():
        #CHANGE
        pred = model(g, pois, costs, indices,gcnOutput, zoneCount)
        loss = criterion(pred, labels)
        meanError = torch.abs((labels - pred.detach()) / labels)
        meanError[meanError > 99999] = 0
        meanError = float(torch.mean(meanError))
        if epoch == 500:
            predictions = pred.detach()
            plt.plot([min(labels), max(labels)], [min(labels), max(labels)], 'r-')
            with open("{}/figures/{}-{}.csv".format(key, split, epoch), mode='w') as wx:
                w = csv.writer(wx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                w.writerow(["#", "label", "estimation"])
                for i in range(len(labels)):
                    w.writerow([i, int(labels[i].item()), int(predictions[i].item())])
            plt.scatter(labels, predictions, facecolors='none', edgecolors='b', s=20, linewidths=0.5)
            plt.title(r'Epoch: {}  |  $R^2$: {}  |  Mean error: {}%'.format(epoch, round(r2_score(predictions, labels), 2),round(meanError, 2)), fontdict={"fontsize":12})
            axes = plt.gca()
            axes.set_xlabel("Real value",fontsize = 12)
            axes.set_ylabel("Estimated value",fontsize = 12)
            axes.set_xlim([min(labels), max(labels)])
            axes.set_ylim([min(labels), max(labels)])
            plt.show(block=False)
            plt.savefig("{}/figures/{}-{}.png".format(key, split, epoch), dpi=300)
            plt.close()
    return [loss.item(), r2_score(pred.detach(), labels), meanError]



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

    pois = torch.zeros((nodeCount,poiCount))
    i = 0
    with open('{}/nodes.csv'.format(key), mode='r') as rx:
        r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in r:
            pois[i][:] = torch.FloatTensor([int(i) for i in row[no:]])
            i += 1

    costs = torch.zeros((zoneCount*zoneCount,1))
    labels = torch.zeros((zoneCount*zoneCount,1))
    count = 0
    with open('{}/costsTrips.csv'.format(key), mode='r') as rx:
        r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in r:
            costs[count][0] = int(row[2])
            labels[count][0] = int(row[3])
            count += 1

    g = dgl.DGLGraph()
    g.add_nodes(nodeCount)


    with open('{}/edges.csv'.format(key), mode='r') as rx:
        r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in r:
            g.add_edge(int(row[0]), int(row[1]))

    print('We have %d nodes.' % g.number_of_nodes())
    print('We have %d edges.' % g.number_of_edges())


    return([g, pois, labels,costs, zoneCount, poiCount])



key = "okokok"

[g, pois, labels,costs, zoneCount, poiCount] = data_collection(key)

with open('{}/indices.csv'.format(key), mode='r') as rx:
    r = csv.reader(rx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    a = [int(i) for i in next(r)]




gcnoutput = 10
models = {}
for i in range(1, 10):
    models[i] = od(poiCount, 64, gcnoutput, 64)


with open('{}/results.csv'.format(key), mode='w') as wx:
    w = csv.writer(wx, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w.writerow(["split", "epoch", "loss_train", "loss_test", "r2", "meanError"])
    for i in range(1,10):
        split = i/10
        breaker = int(split * zoneCount * zoneCount)
        train_indices = a[:breaker]
        test_indices = a[breaker:]
        # (gcninput, gcnhidden, gcnoutput, mlphidden):
        model = models[i]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        for epoch in range(1, 501):
            loss_train = train(optimizer, model, criterion, pois, costs, labels[train_indices], train_indices, zoneCount, gcnoutput)
            [loss_test, r2, meanError] = test(model, criterion, pois, costs, labels[test_indices],test_indices, epoch, zoneCount, gcnoutput, key, split)
            w.writerow([split, epoch, loss_train, loss_test, r2, meanError])
            print(split, epoch, loss_train, loss_test, r2, meanError)
        if i == 8:
            torch.save(model.state_dict(), "{}/model".format(key))
