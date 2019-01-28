import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import random
import time
import json
from os.path import isfile, join
from os import listdir

from torch.utils import data

from NNArchitecture import *

CUDA = True

SAVED_MODELS = 'models'
dataPATH = 'data/'

MAX_EPOCHS = 6
SEQ_LEN = 300
BATCH_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 2
OUTPUT_SIZE = 4
LR = 0.00005


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


def train(model, optimizer, criterion, num_epochs, trainloader, devloader):
    best_dev_loss = 2 ** 20
    big_time = time.time()
    for i in range(num_epochs):
        optimizer.zero_grad()

        j = 1
        train_losses = []
        small_time = time.time()
        for batch, batch_targets in trainloader:
            batch = batch.permute(1, 0, 2)

            y_hat = model(batch.cuda())
            
            
            
            b_targets = []
            for tensor in batch_targets:
                b_targets.append(tensor)
            batch_targets = torch.cat(b_targets)
            
            
            
            loss = criterion(y_hat.cuda(), batch_targets.cuda())
            loss.backward()
            optimizer.step()

            # Progress Bar ######################

            # Percentage
            percent_done = j / len(trainloader)

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
                        (len(trainloader) * i + j) / (len(trainloader) * num_epochs))) - elapsed_in_total

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


#
def catDict(PATH):
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    jsonLists = []

    for file in onlyfiles:
        with open(join(PATH, file)) as fp:
            temp = json.load(fp)
            jsonLists.append(temp)

    #List of tuples of the form: [ ("Standing", {123124.45434: [234, 234, 2343, 454], ... })   , (), ...]
    input_list = []

    for list in jsonLists:
        type = list['type']
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

        input_list.append((type, time_dict))


    return(input_list)


def main():

    inputList = catDict(dataPATH)

    # Rally the Datasets
    train_set_0 = sensorData('train', inputList, SEQ_LEN)
    train_set = torch.utils.data.TensorDataset(train_set_0.sequences, train_set_0.targets.reshape(-1, 1))
    dev_set_0 = sensorData('dev', inputList, SEQ_LEN)
    dev_set= torch.utils.data.TensorDataset(dev_set_0.sequences, dev_set_0.targets.reshape(-1, 1))
    test_set_0 = sensorData('test', inputList, SEQ_LEN)
    test_set= torch.utils.data.TensorDataset(test_set_0.sequences, test_set_0.targets.reshape(-1, 1))


    trainloader = data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    devloader = data.DataLoader(dev_set, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    testloader = data.DataLoader(test_set, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # Define the model, and loss criterion
    input_len = len(inputList[0][1][random.choice(list(inputList[0][1]))])

    device = torch.device("cuda" if CUDA else "cpu")
    model = MyGRU(input_len, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, BATCH_SIZE).to(device)

    criterion = nn.NLLLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train the model
    model = train(model, optimizer, criterion, MAX_EPOCHS, trainloader, devloader)

    #Test the model
    avg_loss = test(model, testloader)

    print("Average Loss on the Test set is: " + str(avg_loss))

main()





