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
        networkActivations = {}
        networkUnactivated = {}
        for data in trainingData:
            scripts = np.array([nr.relu, nr.sigmoid])
            networkActivations[0] = data[1:]
            networkUnactivated[0] = data[1:]
            outputTarget = np.zeros(10)
            outputTarget[int(data[0])] = 1
            layerOutput = data[1:]
            for key, script in zip(wb.keys(), scripts):
                layerOutput, unactivatedOutput = forProp(layerOutput, wb[key]['weights'], wb[key]['biases'], script)
                networkActivations[key] = layerOutput
                networkUnactivated[key] = unactivatedOutput
            
            def backProp(gradients, layerOutput, layerInput, preActInput, layerPos, layerAct):
                outputGradients = []
                for index, grad in enumerate(gradients):
                    wb[layerPos]['weights'][index]-= lr*grad*layerInput
                    wb[layerPos]['biases'][index]-= lr*grad
                    if layerPos > 1:
                        outputGradients.append(grad*np.multiply(layerAct(preActInput),wb[layerPos]['weights'][index]))
                if layerPos <=1:
                    return
                return outputGradients

            dxScripts = np.array([dxSigmoid,dxRelu])
            actualOutput = networkActivations[2]
            expectedOutput = outputTarget
            outputErrors = np.subtract(actualOutput, expectedOutput)
            oGradients = np.multiply(outputErrors,dxSigmoid(actualOutput))        
            networkInput = np.array(networkActivations[0])
            preActZH = np.array(networkUnactivated[1])
            hiddenOutput = np.array(networkActivations[1])
            ogradients = backProp(oGradients, None, hiddenOutput, preActZH, 2, dxRelu)
            for gradient in ogradients:
                backProp(gradient, None, networkInput, None, 1, None)

            '''
            for index, grad in enumerate(oGradients):
                wb[2]['weights'][index]-= lr*grad*hiddenOutput
                wb[2]['biases'][index]-= lr*grad
                hGradients = grad*np.multiply(dxRelu(preActZH),wb[2]['weights'][index])
                for ind, gr in enumerate(hGradients):
                    wb[1]['weights'][ind]-= lr*gr*networkInput
                    wb[1]['biases'][ind]-= lr*gr
            '''


            epochLoss += calcLoss(int(data[0]), layerOutput)
        print('Epoch loss: ', epochLoss/len(trainingData))
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
        trainingData = np.array(trainingData).round(2).tolist()
    start = time()
    #nb.createNetwork(784, (256,), 10, weightRange=[-1,1], biasRange=[0], filename='default.hdf5')
    np.set_printoptions(precision=8, suppress=True, linewidth=200)
    train(10, lr=0.01, trainingData=trainingData)
    #saveModel(nr.readLayers('default.hdf5'))
    end = time()
    print(end-start, "Seconds")