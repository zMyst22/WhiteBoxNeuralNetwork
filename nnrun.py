import nnbuilder as nb
import csv
import numpy as np
import h5py as h5
import time as t

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

def noAct(input):
    return input

def relu(input):
    return np.maximum(0,input)

def readLayers(filename):
    info = {}
    with h5.File(filename, 'r') as nn:
        lkeys = sorted(list(map(lambda x: int(x), list(nn.keys()))))
        for lkey in lkeys:
            info[int(lkey)] = {'weights': np.array(nn[str(lkey)]['weights']), 'biases' : np.array(nn[str(lkey)]['biases'])}
    return info

def runNetwork(filename='default.hdf5', data=np.ndarray, actScript=[None]):
    if type(filename) == str:
        wb = readLayers(filename)
    else:
        wb = filename
    for key, script in zip(wb.keys(), actScript):
        data = activateLayer(data, wb[key]['weights'], wb[key]['biases'], script)

    return np.argmax(data), max(data), data

if __name__ == '__main__':

    with open('mnist_train.csv','r') as f:
       reader = csv.reader(f)
       row = next(reader)
    row = list(map(int, row))
    image = (np.array(row[1:])/255).round(2)

    #nb.createNetwork(784, (64,32), 10, weightRange=[-1,1], biasRange=[-0.5,0.5])
    #print(runNetwork(data=image,actScript=[softmax, relu, sigmoid]))
