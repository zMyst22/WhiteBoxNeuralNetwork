import csv
import numpy as np
import wbnn as nn
from time import time

def forProp(input, weights, bias, activation=None):
    output=np.array([np.sum(np.multiply(input, weight)) for weight in weights])
    output+=bias
    if activation == None:
        return output, output
    else:
        return activation(output), output

def train(epochs=1000, lr=0.1, nnfile='default.hdf5', trainingData=any):
    wb = nn.readLayers(nnfile)
    metadata = wb.pop('metadata')
    scripts, dxScripts = nn.actHandler(metadata['activationScript'])
    totalLoss = 0
    for epoch in range(epochs):
        epochLoss = 0
        for data in trainingData:
            #Left to right
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

            def backProp(gradients, layerInputs, preActInputs, numLayers, dxScripts):
                for index, grad in enumerate(gradients): 
                    wb[numLayers]['weights'][index]-= lr*grad*layerInputs[numLayers-1]
                    wb[numLayers]['biases'][index]-= lr*grad
                    if numLayers > 1:
                        dxActivation = dxScripts[numLayers-2]
                        if dxActivation != None:
                            newGradients = np.array(grad*np.multiply(dxActivation(preActInputs[numLayers-1]),wb[numLayers]['weights'][index]))
                        else:
                            newGradients = np.array(grad*np.multiply(preActInputs[numLayers-1],wb[numLayers]['weights'][index]))
                        backProp(newGradients, layerInputs, preActInputs, numLayers-1, dxScripts)
                    else:
                        pass

            numLayers = len(wb)
            #Left to right
            outputActivation = dxScripts[-1]
            actualOutput = networkActivations[numLayers]
            outputErrors = np.subtract(actualOutput, targetOutput)
            gradients = np.multiply(outputErrors, outputActivation(actualOutput))
            backProp(gradients, networkActivations, networkUnactivated, numLayers, dxScripts)
            epochLoss += nn.calcLoss(int(data[0]), layerOutput)
        print(f'Epoch {epoch+1} - Loss:', epochLoss/len(trainingData))
        totalLoss += epochLoss/len(trainingData)
        
    print('Average loss: ', totalLoss/epochs)
    wb['metadata'] = metadata
    nn.saveModel(wb, 'layer.hdf5')

if __name__ == '__main__': 
    np.set_printoptions(precision=8, suppress=True, linewidth=200)
    #nn.buildNetwork(784, (64,), 10, weightRange=[-0.5,0.5], biasRange=[0], activationScript=['relu', 'sigmoid'], filename='test.hdf5')
    
    trainingData = nn.readMnist('short_mnist.csv')
    start = time()
    train(10, lr=0.01, trainingData=trainingData, nnfile='default.hdf5')
    end = time()
    print(f'Training time: {end-start} seconds')

    start = time()
    nn.testNetwork('layer.hdf5', 'short_mnist_test.csv')
    end = time()
    print(f'Testing time: {end-start} seconds')
    

    