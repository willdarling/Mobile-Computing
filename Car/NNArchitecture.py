
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data




class MyGRU(nn.Module):
        #Output size is # of classification categories
        def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
            super(MyGRU, self).__init__()

            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.num_layers = num_layers

            self.network = nn.GRU(input_size, hidden_size, num_layers)
            self.linear = nn.Linear(hidden_size, output_size)
            self.scores = nn.LogSoftmax(dim=1)

        def forward(self, input):
        
            #print(input.dtype)
        
            output, _ = self.network(input.cuda(), self.initHidden().cuda())
            squeezed = self.linear(output[-1].cuda()).cuda()
            scores = self.scores(squeezed.cuda()).cuda()
            return scores

        def initHidden(self):
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.long)
            return h0



class sensorData(data.Dataset):

    # mode = train/dev/test
    def __init__(self, mode, input_list, sequence_len):
        super(sensorData, self).__init__()

        self.mode = mode

        #[ ("Standing", {123124.45434: [234, 234, 2343, 454], ... })   ,  ()   ]

        # Make a list of sequences and a corresponding list of their target vectors
        sequences = []
        targets = []

        for (action, input_dict) in input_list:

            if 'Jumping' in action:
                targetNum = 0
            elif 'Driving' in action:
                targetNum = 1
            else:
                print("LOOKY HERE ^^^^")
                print(repr(action))
                exit(1)

            keys = sorted(list(input_dict.keys()))

            for i in range(0, len(keys) - sequence_len):

                cur_sequence = []
                for key in keys[i:i + sequence_len]:

                    entry = input_dict[key]
                    cur_sequence.append(entry)

                sequences.append(torch.tensor(cur_sequence, dtype=torch.long))

                targets.append(torch.tensor(targetNum, dtype=torch.long))

        # Define training, development and test sets

        if mode == 'train':

            length = len(sequences)
            lenCheck = len(targets)

            if length != lenCheck:
                print("We have a problem.")
                exit(1)

            sequences = sequences[0:int(length/10)*6] + sequences[int(length/10)*8:]


            targets = targets[0:int(length/10)*6] + targets[int(length/10)*8:]

        if mode == 'dev':

            length = len(sequences)
            lenCheck = len(targets)

            if length != lenCheck:
                print("We have a problem.")
                exit(1)

            sequences = sequences[int(length/10)*6:int(length*7.5/10)]

            targets = targets[int(length/10)*6:int(length*7.5/10)]

        if mode == 'test':

            length = len(sequences)
            lenCheck = len(targets)

            if length != lenCheck:
                print("We have a problem.")
                exit(1)

            sequences = sequences[int(length*7.5/10):int(length/10)*8]
            targets = targets[int(length*7.5/10):int(length/10)*8]

        self.sequences = torch.LongTensor(torch.stack(sequences))
        self.targets = torch.LongTensor(torch.stack(targets))

    def __len__(self):
        'Returns the total number of samples'
        return(len(self.targets))

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.sequences[index], self.targets[index]


