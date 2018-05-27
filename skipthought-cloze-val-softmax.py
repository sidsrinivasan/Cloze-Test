import random
import preproc
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import time
import math

bestAcc = 0.0

use_cuda = torch.cuda.is_available()
TEST_SET_SIZE = 80000

SAVE_FILE = 'relu-skip-rel-soft-val'
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=2):
        """
            FC-ReLU-FC-ReLU-FC-FC-Softmax
        """
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size

        self.layer1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.layer2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.layer3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        nn.init.xavier_uniform(self.layer1.weight.data)
        nn.init.xavier_uniform(self.layer2.weight.data)
        nn.init.xavier_uniform(self.layer3.weight.data)
        nn.init.xavier_uniform(self.output_layer.weight.data)

    def forward(self, input):
        output = self.relu(self.layer1(input))
        output = self.relu(self.layer2(output))
        output = self.layer3(output)
        output = self.output_layer(output)
        output = output.view(output.size()[0],-1)
        return self.softmax(output)


def train(input_variable, target_variable, MLP, MLP_optimizer, criterion):
    """
        Train the MLP on a batch
    """

    MLP_optimizer.zero_grad()

    loss = 0
    batch_size = input_variable.size()[0]
    prediction = MLP(input_variable)
    loss = criterion(prediction, target_variable)

    loss.backward()
    MLP_optimizer.step()

    return loss.data[0] / batch_size


# Helper functions to print time elapsed and remaining
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(MLP, n_iters, batch_size=10, print_every=1000, plot_every=100, sample_every=1000, \
                learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  
    plot_loss_total = 0 

    plot_test_losses = []
    plot_test_acc = []

    # Initialize optimizer and loss
    MLP_optimizer = optim.SGD(MLP.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    evaluate(MLP, batch_size, criterion)
    print('STARTING TRAINING...')

    for iter in range(1, n_iters + 1):
        # create pyTorch Variable of batch to train
        training_pair = preproc.stitchVariables(random.sample(training_pairs, batch_size))
        input_variable = Variable(training_pair[0].cuda())
        target_variable = Variable(training_pair[1].cuda())

        # Train on batch and get loss
        loss = train(input_variable, target_variable, MLP, MLP_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # Status updates
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter *1.0 / n_iters),
                                         iter, iter *1.0 / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter % sample_every == 0:
            print('\n-------------Random Sampling-------------\n')
            plot_test_losses += [evaluate(MLP, batch_size, criterion)]
            plot_test_acc += [getAccuracy(MLP, batch_size)]
            print('\n------------------DONE------------------\n')
    savePlot(plot_losses, 'Relevance_TrainingLoss.jpg')
    savePlot(plot_test_losses, 'Relevance_ValLoss.jpg')
    savePlot(plot_test_acc, 'Relevance_ValAccuracy.jpg')


def savePlot(points, filename):
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(points)
    fig.savefig(filename)


def evaluate(MLP, batch_size, criterion):
    """
        Evaluate Loss on the 'testing_pairs' set
        Note: This is NOT a direct measure of performance on cloze test; to have low loss,
              model must predict the 'wrong' ending as 0, and the right 'ending' as 1. In practice,
              this is not required as both endings may fit (i.e., label '1'), but the 'right' ending
              is more right than the 'wrong' ending. See getAccuracy() for details.
    """
    loss = 0.0
    for i in range(0, testing_pairs[0].size()[0], batch_size):
        input_variable1 = Variable(testing_pairs[0][i:i+batch_size].cuda())
        input_variable2 = Variable(testing_pairs[1][i:i+batch_size].cuda())

        target_variable1 = Variable(testing_pairs[2][i:i+batch_size] == 0).type(torch.LongTensor).cuda()
        target_variable2 = Variable(testing_pairs[2][i:i+batch_size] == 1).type(torch.LongTensor).cuda()

        prediction1 = MLP(input_variable1)
        prediction2 = MLP(input_variable2)

        loss += (criterion(prediction1, target_variable1) + criterion(prediction2, target_variable2)) / 2.0

    print("Val Loss: %f" % (loss.data[0] /testing_pairs[0].size()[0]))
    return loss.data[0] / testing_pairs[0].size()[0]


def getRegularAccuracy(MLP_cloze, batch_size):
    """ 
        Measure of accuracy NOT on the Cloze Test; for all the samples in testing_pairs, count the
        number of times the right endings were predicted as right and the wrong endings were predicted
        as wrong, and present as a ratio of total samples.
    """
    
    tally = 0
    mispred_mat = np.zeros([2,2])
    target_tallies = np.zeros([1,2])
    pred_tallies = np.zeros([2,1])

    for i in range(0, testing_pairs[0].size()[0], batch_size):
        input_variable1 = Variable(testing_pairs[0][i:i+batch_size].cuda())
        input_variable2 = Variable(testing_pairs[1][i:i+batch_size].cuda())

        target_variable1 = Variable(testing_pairs[2][i:i+batch_size] == 0).type(torch.LongTensor).cuda()
        target_variable2 = Variable(testing_pairs[2][i:i+batch_size] == 1).type(torch.LongTensor).cuda()

        prediction1 = MLP_cloze(input_variable1)
        probs, predictions1 = torch.max(prediction1.data, 1)

        prediction2 = MLP_cloze(input_variable2)
        probs, predictions2 = torch.max(prediction2.data, 1)

        tally += np.sum(predictions1.cpu().numpy() == target_variable1.data.cpu().numpy()) + \
        np.sum(predictions2.cpu().numpy() == target_variable2.data.cpu().numpy())

        for j in range(2):
            for k in range(2):
                mispred_mat[j,k] += torch.sum((predictions1 == j) + (target_variable1.data == k) == 2) + \
                                    torch.sum((predictions2 == j) + (target_variable2.data == k) == 2)
            target_tallies[0,j] += torch.sum(target_variable1.data == j) + torch.sum(target_variable2.data == j)
            pred_tallies[j] += torch.sum(predictions1 == j) + torch.sum(predictions2 == j)

    acc = tally*0.5/(testing_pairs[0].size()[0])
    print("Val Accuracy: %f" % acc)
    print("Prediction Matrix as Ratio of Targets: ")
    print(mispred_mat/target_tallies)
    print("Prediction Matrix as Ratio of Predictions: ")
    print(mispred_mat/pred_tallies)

    return acc

def getAccuracy(MLP, batch_size):
    """ 
        Measure of accuracy on the Cloze Test.
        For each set of 4 sentences, present sentence 5 and sentence 6 as options, and
        pick the sentence that is more likely (according to softmax) as the 'right' ending.
    """
    
    tally = 0

    for i in range(0, testing_pairs[0].size()[0], batch_size):
        input_variable1 = Variable(testing_pairs[0][i:i+batch_size].cuda())
        input_variable2 = Variable(testing_pairs[1][i:i+batch_size].cuda())

        prediction1 = MLP(input_variable1)[:,1]
        prediction2 = MLP(input_variable2)[:,1]
        predictions = prediction1 < prediction2

        target_variable = Variable(testing_pairs[2][i:i+batch_size].cuda())
        tally += np.sum(predictions.data.cpu().numpy() == target_variable.data.cpu().numpy())

    acc = tally*1.0/(testing_pairs[0].size()[0])
    print("Val Accuracy: %f" % acc)

    # global bestAcc
    # if acc > bestAcc:
    #     bestAcc = acc
    #     torch.save(MLP, SAVE_FILE + str(acc)[:6])
    return acc


if __name__ == '__main__':

    # Gather data
    f1 = h5py.File('ROC_encoded.hdf5', 'r')
    f2 = h5py.File('ROCval_encoded.hdf5', 'r')
    f3 = h5py.File('ROCtest_encoded.hdf5', 'r')
    
    stories, training_pairs, val_pairs, test_pairs = preproc.get_data(f1, f2, f3)

    print('Training Samples: %i' % len(training_pairs))
    print('Val Samples: %i' % len(val_pairs))
    print('Test Samples: %i' % len(test_pairs))

    # Initialize Model
    hidden_sizes = [2400, 1200, 600]
    mlp_cloze = MLP(4800, hidden_sizes)

    if use_cuda:
        mlp_cloze = mlp_cloze.cuda()

    # Prepare validation set as torch Variables
    print("Encoding Val Data...")
    testing_pairs = preproc.stitchVariables_clozetest(test_pairs)
    print("DONE!")

    mlp_cloze = torch.load('skip-rel-soft-val07553.pt')
    getAccuracy(mlp_cloze, 20)
    # trainIters(mlp_cloze, 100000, print_every=100, plot_every=500, sample_every=5000)

    f1.close()
    f2.close()
    f3.close()
