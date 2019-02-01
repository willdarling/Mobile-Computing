import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import random
import time
import json
import ast
import math
from random import shuffle
from os.path import isfile, join
from os import listdir

from torch.utils import data

from NNArchitecture import *

CUDA = True

SAVED_MODELS = 'models'
dataPATH = 'data/'

SEGDIVS = 10

MAX_EPOCHS = 60
SEQ_LEN = 1000
BATCH_SIZE = 512
HIDDEN_SIZE = 64
NUM_LAYERS = 4
OUTPUT_SIZE = 4
LR = 0.001


def saveModel(model, dev_loss, PATH):
    with open(PATH + "/model_" + str(dev_loss.item()), 'x') as fp:
        fp.write("MAX_EPOCHS = 300\nSEQ_LEN = 120\nBATCH_SIZE = 30")
        fp.write("HIDDEN_SIZE = 240\nNUM_LAYERS = 3\nOUTPUT_SIZE = 3")  # TODO Update this
        fp.write("LR = 0.00005")
    torch.save(model.state_dict(), PATH + '/' + str(dev_loss))


def loadModel(PATH):
    model = MyGRU(input_len, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, BATCH_SIZE)
    model.load_state_dict(torch.load(PATH))

    return model


def intToStr(i):
    if i < 10:
        return "0" + str(int(i))
    else:
        return str(int(i))


def formatSeconds(seconds):
    days = seconds // (60 * 60 * 24)
    seconds -= days * (60 * 60 * 24)
    hours = seconds // (60 * 60)
    seconds -= hours * (60 * 60)
    minutes = seconds // 60
    seconds -= minutes * 60
    seconds = int(seconds)

    days = intToStr(days)
    hours = intToStr(hours)
    minutes = intToStr(minutes)
    seconds = intToStr(seconds)

    return days + ":" + hours + ":" + minutes + ":" + seconds


def train(model, optimizer, criterion, num_epochs, trainList, devList):
    best_dev_loss = 2 ** 20
    big_time = time.time()
    k = 0


    devloader = loaderMaker(devList, 0, len(devList))

    
    for i in range(num_epochs):
        optimizer.zero_grad()

        j = 1
        train_losses = []
        small_time = time.time()
        
        n = 0
        while n / SEGDIVS < 1:
         
            curStart = n * int((len(trainList)/SEGDIVS))
            curEnd = curStart + int((len(trainList)/SEGDIVS))
            
            if curEnd >= len(trainList):
                curEnd = int((len(trainList)/SEGDIVS)) - 1

            trainloader = loaderMaker(trainList, curStart, curEnd)


            for batch, batch_targets in trainloader:
                k  += 1
                
  
                batch = batch.permute(1, 0, 2)
                

                optimizer.zero_grad()

                y_hat = model(batch.cuda())
            
                b_targets = []
                for tensor in batch_targets:
                    b_targets.append(tensor)
                batch_targets = torch.cat(b_targets)
                
                
                
                loss = criterion(y_hat.cuda(), batch_targets.cuda())
                loss.backward()
                optimizer.step()


                """

                #Cosine annealing of the learning rate
                batches_per_epoch = len(trainloader)
                sum_total = batches_per_epoch * MAX_EPOCHS
                batches_seen = (batches_per_epoch * i) + k
                
                
                
                

                lr = LR * ((1 + math.cos(math.pi * batches_seen / sum_total)) / 2)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                """
                
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                norms = []
                total_norm = 0

                for p in model.parameters():
                    norm = p.grad.data.norm()

                    if norm > 0.25:
                        p.grad.data.div_(max(norm, 1e-6) / 0.25)
                        
                        
                        
                        

                # Progress Bar ######################

                # Percentage
                percent_done = j / (len(trainloader) * SEGDIVS)

                tens_ones = str(int(percent_done * 100))
                tenths_hundths = str(int((percent_done % 0.01) * 10000))

                if len(tenths_hundths) < 2:
                    tenths_hundths = tenths_hundths + "0"

                percent_done_epoch = tens_ones + '.' + tenths_hundths

                # Time
                elapsed_in_epoch = time.time() - small_time
                elapsed_in_total = time.time() - big_time

                time_left_epoch = (elapsed_in_epoch / percent_done) - elapsed_in_epoch
                time_left_total = (elapsed_in_total / (
                            (len(trainloader) * i * SEGDIVS + j) / (len(trainloader) * SEGDIVS * num_epochs))) - elapsed_in_total

                time_left_epoch = formatSeconds(time_left_epoch)
                time_left_total = formatSeconds(time_left_total)

                print(percent_done_epoch + "% through Epoch " + str(i + 1) + "/" + str(num_epochs), end='')
                print(" | Time Left: " + time_left_epoch + " | Total Time Left: " + time_left_total, end='\r')

                #######################################

                j += 1

            dev_loss = test(model, devloader)
            accuracy = 1 / np.exp(dev_loss)
            print("-" * 50)
            print("Train Loss: " + str(loss.item()) + ", Train Accuracy: " + str(1 / np.exp(loss.item())) + "\n")
            print("Dev Loss:   " + str(dev_loss) + ", Dev Accuracy:    " + str(accuracy))

            if dev_loss < best_dev_loss:
                print("[New Best. Saving Model...]\n")
                saveModel(model, dev_loss, SAVED_MODELS)
                best_dev_loss = dev_loss

            
            n += 1

    return model


def test(model, testloader):
    losses = []
    criterion = nn.NLLLoss()

    for batch, batch_targets in testloader:
        batch = batch.permute(1, 0, 2)
        
        b_targets = []
        for tensor in batch_targets:
            b_targets.append(tensor)
        batch_targets = torch.cat(b_targets)


        y_hat = model(batch.cuda())
        loss = criterion(y_hat.cuda(), batch_targets.cuda())
        losses.append(loss.item())

    avg_loss = np.mean(losses)

    return avg_loss


"""
def dataSlicer(inputList, start, end):

    slice = inputList[start:end]
    
    targets = []
    majorSequences = []
    
    for tuple in slice:
        target = tuple[0]
        mSeq = tuple[1]
        
        majorSequences.append(mSeq)
        targets.append(target)
        
    return (majorSequences, targets)
"""


def loaderMaker(inputList, start, end):

    input = inputList[start:end]

    inputSet = sensorData(input, SEQ_LEN)
    inputLoaderSet = torch.utils.data.TensorDataset(inputSet.sequences, inputSet.targets.reshape(-1, 1))
    

    inputLoader = []
    
    for i in range(0, int(len(inputLoaderSet)/BATCH_SIZE)-1):

        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        
        if end >= len(inputLoaderSet):
            end = len(inputLoaderSet) - 1

        inputLoader.append((inputLoaderSet[start:end][0], inputLoaderSet[start:end][1]))
  
    return inputLoader



def catDict(PATH):
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    jsonLists = []

    for file in onlyfiles:
        with open(join(PATH, file)) as fp:
            temp = json.load(fp)
            jsonLists.append(temp)

    #List of tuples of the form: [ (2, {123124.45434: [234, 234, 2343, 454], ... })   , (), ...]
    input_list = []

    for list in jsonLists:
        for dict in list:
            action = dict['type']
            
            if 'Jumping' in action:
                targetNum = 0
            elif 'Driving' in action:
                targetNum = 1
            elif 'Standing' in action:
                targetNum = 2
            elif 'Walking' in action:
                targetNum = 3
            else:
                print("LOOKY HERE ^^^^")
                print(repr(action))
                exit(1)
            
            
            time_dict = {}
            for sample in dict['seq']:

                time = sample["time"]

                sample = sample['data']

                xGyro = float(sample['xGyro'])
                zAccl = float(sample['zAccl'])
                yGyro = float(sample['yGyro'])
                zGyro = float(sample['zGyro'])
                xAccl = float(sample['xAccl'])
                xMag = float(sample['xMag'])
                yMag = float(sample['yMag'])
                zMag = float(sample['zMag'])
                yAccl = float(sample['yAccl'])

                time_dict[float(time)] = [xGyro, zAccl, yGyro, zGyro, xAccl, xMag, yMag, zMag, yAccl]

            input_list.append((targetNum, time_dict))


    return(input_list)



#
def catDictOld(PATH):
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    jsonLists = []

    for file in onlyfiles:
        with open(join(PATH, file)) as fp:
            temp = json.load(fp)
            jsonLists.append(temp)

    #List of tuples of the form: [ ("Standing", {123124.45434: [234, 234, 2343, 454], ... })   , (), ...]
    input_list = []

    for list in jsonLists:
        action = list['type']
        
        if 'Jumping' in action:
                targetNum = 0
        elif 'Driving' in action:
            targetNum = 1
        elif 'Standing' in action:
            targetNum = 2
        elif 'Walking' in action:
            targetNum = 3
        else:
            print("LOOKY HERE ^^^^")
            print(repr(action))
            exit(1)
        
        
        time_dict = {}
        for sample in list['seq']:

            time = sample["time"]

            sample = sample['data']

            xGyro = float(sample['xGyro'])
            zAccl = float(sample['zAccl'])
            yGyro = float(sample['yGyro'])
            zGyro = float(sample['zGyro'])
            xAccl = float(sample['xAccl'])
            xMag = float(sample['xMag'])
            yMag = float(sample['yMag'])
            zMag = float(sample['zMag'])
            yAccl = float(sample['yAccl'])

            time_dict[float(time)] = [xGyro, zAccl, yGyro, zGyro, xAccl, xMag, yMag, zMag, yAccl]

        input_list.append((targetNum, time_dict))


    return(input_list)


def main():

    """ DATA PROCESSING
    inputList = catDict(dataPATH)
    inputListTest = catDictOld(dataPATH + "test/")
    
    shuffle(inputList)
    
    
    json.dump(inputList, open('shuffledWhole.json', 'w'))
    exit(1)
    """
    
    with open('shuffledWhole.json', 'r') as fp:
        inputList = ast.literal_eval(fp.read())
        
        
    length = len(inputList)
    
    testList = inputList[length-10:]
    devList = inputList[length-25:length-10]
    trainList = inputList[:length-25]

    # Rally the Datasets

    test_set_0 = sensorData(testList, SEQ_LEN)
    test_set= torch.utils.data.TensorDataset(test_set_0.sequences, test_set_0.targets.reshape(-1, 1))


    
    testloader = data.DataLoader(test_set, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # Define the model, and loss criterion
    input_len = len(inputList[0][1][random.choice(list(inputList[0][1]))])

    device = torch.device("cuda" if CUDA else "cpu")
    model = MyGRU(input_len, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, BATCH_SIZE).to(device)

    criterion = nn.NLLLoss()

    # Define the optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay = 0.1)

    # Train the model
    model = train(model, optimizer, criterion, MAX_EPOCHS, trainList, devList)

    #Test the model
    avg_loss = test(model, testloader)
    
    accuracy = 1 / np.exp(avg_loss)

    print("Average Loss on the Test set is: " + str(avg_loss) + ", Accuracy: " + str(accuracy))

main()





