import h5py as h5
import nnbuilder as nb
import nnrun as nr
import numpy as np
import csv
from time import time

def forProp(input, weights, bias, activation=None):
    output=np.array([np.sum(np.multiply(input, weight)) for weight in weights])
    output+=bias
    if activation == None:
        return output, output
    else:
        return activation(output), output

def calcLoss(target, outputLayer):
    targets = np.zeros(len(outputLayer))
    targets[target] = 1
    return np.sum(np.subtract(outputLayer, targets)**2 )

def dxSigmoid(input):
    num = nr.sigmoid(input)
    return num*(1-num)

def dxRelu(input):
    return np.where(np.array(input) > 0, 1, 0)

def train(epochs=1000, lr=0.1, nnfile='default.hdf5', trainingData=any):
    wb = nr.readLayers(nnfile)
    totalLoss = 0
    for epoch in range(epochs):
        epochLoss = 0
        for data in trainingData:
            scripts = np.array([nr.relu, nr.sigmoid])
            networkActivations = {}
            networkUnactivated = {}
            networkActivations[0] = data[1:]
            networkUnactivated[0] = data[1:]
            targetOutput = np.zeros(10)
            targetOutput[int(data[0])] = 1
            layerOutput = data[1:]
            for key, script in zip(wb.keys(), scripts):
                layerOutput, unactivatedOutput = forProp(layerOutput, wb[key]['weights'], wb[key]['biases'], script)
                networkActivations[key] = layerOutput
                networkUnactivated[key] = unactivatedOutput

            def backProp(gradients, layerInput, preActInput, layerPos, layerAct):
                newGradients = []
                for index, grad in enumerate(gradients):
                    wb[layerPos]['weights'][index]-= lr*grad*layerInput
                    wb[layerPos]['biases'][index]-= lr*grad
                    if layerPos > 1:
                        newGradients.append(grad*np.multiply(layerAct(preActInput),wb[layerPos]['weights'][index]))
                if layerPos <=1:
                    return None
                return newGradients
        
            numLayers = len(wb)
            dxScripts = [dxSigmoid, dxRelu, None]
            outputActivation = dxScripts.pop(0)
            actualOutput = networkActivations[numLayers]
            outputErrors = np.subtract(actualOutput, targetOutput)
            gradients = np.multiply(outputErrors, outputActivation(actualOutput))

            gradients = backProp(gradients, networkActivations[1], networkUnactivated[1], 2, dxRelu)
            for grads in gradients:
                backProp(np.array(grads),networkActivations[0], networkUnactivated[0], 1, None)

            epochLoss += calcLoss(int(data[0]), layerOutput)
        print(f'Epoch {epoch} - Loss:', epochLoss/len(trainingData))
        totalLoss += epochLoss/len(trainingData)
        
    print('Average loss: ', totalLoss/epochs)
    saveModel(wb, 'layer.hdf5')
    
def saveModel(model:dict, modelName='default.hdf5'):
    with h5.File(modelName, 'w') as m:
        for key in model.keys():
            m.create_group(str(key))
            wgrp = m[str(key)]
            wgrp['weights'] = model[key]['weights']
            wgrp['biases'] = model[key]['biases']
        print(f'Model saved as {modelName}')
    return

if __name__ == '__main__':
    with open('short_mnist.csv','r') as f:
        def setUp(row=np.ndarray):
                row = np.array(row, dtype=float)
                label = row[0]
                row/=255
                row[0] = label
                return row  
        
        reader = csv.reader(f)
        trainingData = list(map(setUp, reader))
        trainingData = np.array(trainingData).round(2)

    start = time()
    #nb.createNetwork(784, (32,32), 10, weightRange=[-1,1], biasRange=[0], filename='test.hdf5')
    np.set_printoptions(precision=8, suppress=True, linewidth=200)
    train(10, lr=0.01, trainingData=trainingData, nnfile='default.hdf5')
    #saveModel(nr.readLayers('default.hdf5'))
    end = time()
    print(end-start, "Seconds")