import nnbuilder as nb
import csv
import numpy as np
import h5py as h5
import time as t
import zverter as z

def activateLayer(input, weights, bias, activation=None):
    output=np.array([sum(input * weight) for weight in weights])
    output+=bias
    if activation == None:
        return output
    else:
        return activation(output)

def softmax(input):
    output = (input-min(input))/(max(input)-min(input))
    return output

def sigmoid(input):
    output = 1/(1+np.exp(-input))
    return output

def relu(input):
    return np.maximum(0,input)

def readLayers(filename):
    info = {}
    with h5.File(filename, 'r') as nn:
        weights = nn['weights']
        biases = nn['biases']
        wkeys = sorted(list(nn['weights'].keys()), key=lambda x: int(x[1:]))
        bkeys = sorted(list(nn['biases'].keys()), key=lambda x: int(x[1:]))

        for weight, bias in zip(wkeys, bkeys):
            info[int(weight[1:])] = {'w':np.array(weights[weight]), 'b':np.array(biases[bias])}
    return info

def runNetwork(filename='default.hdf5', data=np.ndarray):
    wb = readLayers(filename)
    actScript = [softmax, relu]
    for key, script in zip(wb.keys(), actScript):
        data = activateLayer(data, wb[key]['w'], wb[key]['b'], script)

    print(np.argmax(data))

with open('mnist_train.csv','r') as f:
    reader = csv.reader(f)
    row = next(reader)
row = list(map(int, row))
image = (np.array(row[1:])/255).round(2)

nb.createNetwork(784, (32,16), 10, weightRange=[-1,1], biasRange=[-0.5,0.5])
runNetwork(data=image)