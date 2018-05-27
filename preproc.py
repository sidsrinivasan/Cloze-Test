import numpy as np
import random
import h5py
import sys

import torch
from torch.autograd import Variable

NEG_LABEL = 0
RHO = 0.95

use_cuda = torch.cuda.is_available()


def get_data(f1, f2, f3):

    # Collect skip-thought encodings
    train_stories = torch.from_numpy(f1['fullROC'][:]).type(torch.FloatTensor)

    val_stories = torch.from_numpy(f2['fullROC_val'][:]).type(torch.FloatTensor)
    val_labels = torch.from_numpy(f2['fullROC_labels'][:]).type(torch.LongTensor)

    test_stories = torch.from_numpy(f3['fullROC_val'][:]).type(torch.FloatTensor)
    test_labels = torch.from_numpy(f3['fullROC_labels'][:]).type(torch.LongTensor)

    print("DONE compiling stories!")

    # Shuffle stories
    np.random.seed(0)
    order = list(np.random.permutation(np.arange(val_stories.shape[0]-1)))

    trainingStoriesIndices = order[:int(RHO * len(order))]
    valStoriesIndices = order[int(RHO * len(order)):]
    testStoriesIndices = range(test_stories.shape[0])


    sys.stdout.write("DONE!\nCompiling Training Data...")
    sys.stdout.flush()

    # Compile dict of test set with focus event and context pairs
    trainingSet = []

    # Positive Examples from ROC train set
    # for index, t in enumerate(range(train_stories.size()[0])):
    #     if index % 1000 == 0:
    #         print('Compiled %i stories!' % index)
    #     story = train_stories[t]
    #     # ordering_neg = torch.cat([story[3], story[4]])
    #     ordering_pos = story[3] + story[4]

    #     trainingSet += [(ordering_pos, 1)]

    # Positive and Negative Examples from ROC val set
    for index, t in enumerate(trainingStoriesIndices):
        if index % 1000 == 0:
            print('Compiled %i stories!' % index)
        story = val_stories[t]

        if val_labels[t]:
            # ordering_pos = torch.cat([story[3], story[5]])
            # ordering_neg = torch.cat([story[3], story[4]])

            ordering_pos = story[3] + story[5]
            ordering_neg = story[3] + story[4]
        else:

            # ordering_pos = torch.cat([story[3], story[4]])
            # ordering_neg = torch.cat([story[3], story[5]])

            ordering_pos = story[3] + story[4]
            ordering_neg = story[3] + story[5]

        trainingSet += [(ordering_pos, 1), (ordering_neg, NEG_LABEL)]

    # Shuffle test set by dict keys
    np.random.shuffle(trainingSet)
    np.random.seed(None)


    sys.stdout.write("DONE!\nCompiling Val Data...")
    sys.stdout.flush()
    
    # Positive and negative examples from ROC val data -- held out val set
    valSet = []
    for index, t in enumerate(valStoriesIndices):
        if index % 1000 == 0:
            print('Compiled %i stories!' % index)
        story = val_stories[t]

        # ordering_pos = torch.cat([story[3].view(1,1,-1), story[4].view(1,1,-1)], 2)
        # ordering_neg = torch.cat([story[3].view(1,1,-1), story[5].view(1,1,-1)], 2)

        ordering_pos = (story[3] + story[4]).view(1,1,-1)
        ordering_neg = (story[3] + story[5]).view(1,1,-1)

        valSet += [(ordering_pos, ordering_neg, (NEG_LABEL,1)[val_labels[t]])]

    # Shuffle test set by dict keys
    np.random.shuffle(valSet)
    np.random.seed(None)


    sys.stdout.write("DONE!\nCompiling Test Data...")
    sys.stdout.flush()

    # Positive and negative examples from ROC test data
    testSet = []
    for index, t in enumerate(testStoriesIndices):
        if index % 1000 == 0:
            print('Compiled %i stories!' % index)
        story = test_stories[t]
        
        # ordering_pos = torch.cat([story[3].view(1,1,-1), story[4].view(1,1,-1)], 2)
        # ordering_neg = torch.cat([story[3].view(1,1,-1), story[5].view(1,1,-1)], 2)

        ordering_pos = (story[3] + story[4]).view(1,1,-1)
        ordering_neg = (story[3] + story[5]).view(1,1,-1)

        testSet += [(ordering_pos, ordering_neg, (NEG_LABEL,1)[test_labels[t]])]

    # Shuffle test set by dict keys
    np.random.shuffle(testSet)
    np.random.seed(None)

    sys.stdout.write("DONE!\n")
    sys.stdout.flush()

    return test_stories, trainingSet, valSet, testSet


def stitchVariables(sample):
    """
        Given a list of 2-tuples, where each tuple is 
        (torch.FloatTensor input, int target), return a concatenated
        FloatTensor so you can batch it.

        Note: move to GPU and make variable during runtime
    """
    input_Var = torch.cat([s[0].view(1,1,-1) for s in sample])
    target_Var = torch.cat([torch.LongTensor([s[1]]) for s in sample])

    return (input_Var, target_Var)


def stitchVariables_clozetest(sample):
    """
        Given a list of 3-tuples, where each tuple is 
        (torch.FloatTensor input1, torch.FloatTensor input2, int correct_label),
        return a concatenated FloatTensor so you can batch it.

        Note: move to GPU and make variable during runtime
    """
    input_Var1 = torch.cat([s[0].view(1,1,-1) for s in sample])
    input_Var2 = torch.cat([s[1].view(1,1,-1) for s in sample])
    target_Var = torch.cat([torch.LongTensor([s[2]]) for s in sample])

    return (input_Var1, input_Var2, target_Var)