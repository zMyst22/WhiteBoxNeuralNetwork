import csv
import h5py as h5
import numpy as np
from PIL import Image

###Activation Funtions followed by their derivatives###

def relu(input):
    return np.maximum(0,input)

def dxRelu(input):
    return np.where(np.array(input) > 0, 1, 0)

def sigmoid(input):
    output = 1/(1+np.exp(-input))
    return output 

def dxSigmoid(input):
    num = sigmoid(input)
    return num*(1-num)

###Common Functions###

#Converts strings into function objects
def actHandler(scripts:list):
    #When adding new activation functions, define them above and add them to the dictionary below in the same format
    #Keep the strings all lower case
    scriptsDict = {'relu' : relu,
                   'dxrelu' : dxRelu,
                   'sigmoid' : sigmoid,
                   'dxsigmoid' : dxSigmoid,
                   'none' : None,
                   'dxnone' : None}
    actScript = []
    dxActScript = []
    for script in scripts:
        actScript.append(scriptsDict[script])
        dxActScript.append(scriptsDict[f'dx{script}'])
    return tuple(actScript), tuple(dxActScript)

#Performs an entire layer's activation. Taking in an input, all of that layer's weights and biases then applies the activation function
def activateLayer(input, weights, bias, activation=None):
    output=np.array([sum(input * weight) for weight in weights])
    output+=bias
    if activation == None:
        return output
    else:
        return activation(output)
    
#Initializes a network with randoms weights and automatically creates the correct number of layers, neurons, weights, and biases. 
#Randomizing starting biases is not common practice, but the functionality is there to try, although it is recommended to keep at default.
def buildNetwork(inputSize:int, hiddenLayers:tuple, outputSize:int, activationScript:list, filename='default.hdf5', weightRange=[-1,1], biasRange=[0], precision=2):
    if hiddenLayers == None:
        hiddenLayers = []
        layers =[inputSize]+[outputSize]
    else:
        layers = [inputSize]+list(hiddenLayers)+[outputSize]
    
    if len(biasRange) == 1:
        biasMin, biasMax = biasRange[0], biasRange[0]
    else:
        biasMin, biasMax = biasRange
    #Sidenote - This code is just elegant af 
    with h5.File(filename, 'w') as nn:
        for layer in range(len(layers)-1):
            weights = np.random.uniform(weightRange[0],weightRange[1],(layers[layer+1],layers[layer])).round(precision)
            biases = np.random.uniform(biasMin, biasMax,(layers[layer+1],)).round(precision)
            lgrp = nn.create_group(str(layer+1))
            wset = lgrp.create_dataset('weights', data=weights)
            bset = lgrp.create_dataset('biases', data=biases)
        mgrp = nn.create_group('metadata')
        aset = mgrp.create_dataset('activationScript', data=activationScript, shape=len(activationScript), dtype=h5.string_dtype())
        print(f'File saved as: {filename}')
        print(f'Network has {inputSize} input neuron(s), {len(hiddenLayers)} hidden layer(s), and {outputSize} output neuron(s).')
        print(f'The initial weight range is between {weightRange[0]} and {weightRange[1]}.')
        if len(biasRange)==1:
            print(f'Biases have been initialized at {biasRange[0]}.')
        else:
            print(f'Biases have been randomly initialized between {biasRange[0]} and {biasRange[1]}')

#Calculates the loss between the actual output and the expected output, on the scale of an entire layer
def calcLoss(target, outputLayer):
    targets = np.zeros(len(outputLayer))
    targets[target] = 1
    return np.sum(np.subtract(outputLayer, targets)**2 )

def parseImage(layer):
    print(len(layer))
    data = np.array(layer)
    data -= np.min(data)
    data /= np.max(data)
    data *= 255
    neuron = []
    for i in range(16):
        j = i*16
        neuron.append(data[j:j+16])
    neuron = np.array(neuron)
    return neuron.astype(int)

#Takes in a numpy array and pa
def parseImages(layer):
    neurons = []
    for line in layer:
        neuron = []
        for i in range(28):
            j = i*28
            neuron.append(line[j:j+28])
        neurons.append(neuron)
    image = []
    for i in range(16):
        j = i * 16
        image.append(neurons[j:j+16])
    data = []
    lowest = np.min(layer)
    highest = np.max(layer)
    for q in range(16):
        for n in range(28):
            row = []
            for m in range(16):
                row.extend(image[q][m][n])
                row.extend([lowest])
            data.append(row)
        data.append(np.full(len(row),lowest))
    
    data = np.array(data)
    data -= np.min(data)
    data /= np.max(data)
    data *= 255
    return data.astype(int)

#Takes a HDF5 file and converts the data into a dictionary of Numpy arrays
def readLayers(filename):
    info = {}
    with h5.File(filename, 'r') as nn:
        info['metadata'] = {'activationScript': [script.decode() for script in nn['metadata']['activationScript']]}
        keys = list(nn.keys())
        keys.remove('metadata')
        lkeys = sorted(list(map(lambda x: int(x), keys)))
        for lkey in lkeys:
            info[int(lkey)] = {'weights': np.array(nn[str(lkey)]['weights']), 'biases' : np.array(nn[str(lkey)]['biases'])}
    return info
    
#Converts MNIST CSV data into a format our network can use. Used for pulling the training data as well as the testing data
def readMnist(csvfile):
    with open(csvfile,'r') as f:
        #setUp's only purpose is to format the data from the CSVs, therefore is childed here
        def setUp(row=np.ndarray):
                row = np.array(row, dtype=float)
                label = row[0]
                row/=255
                row[0] = label
                return row  
        
        reader = csv.reader(f)
        dirtyData = list(map(setUp, reader))
        cleanData = np.array(dirtyData).round(2)
    return cleanData

#Stitches together our readLayers and activateLayer functions. filename can either be an actual file name or can be a dictionary if readLayers was used previously
def runNetwork(filename='default.hdf5', data=np.ndarray, actScript=None):
    if type(filename) == str:
        wb = readLayers(filename)
    else:
        wb = filename
    if actScript == None:
        actScript = actHandler(wb['metadata']['activationScript'])
    keys = list(wb.keys())
    keys.remove('metadata')
    for key, script in zip(keys, actScript):
        data = activateLayer(data, wb[key]['weights'], wb[key]['biases'], script)

    return np.argmax(data), max(data), data

#Takes in parsed image data and saves it as a png
def saveImage(imgArray, filename):
    image_array = np.array(imgArray, dtype=np.uint8)
    image = Image.fromarray(image_array)
    image.save(filename)
    return

#Saves the network into an HDF5 model, with model being the network and modelName being the file name to be saved under
def saveModel(model:dict, modelName='default.hdf5'):
    with h5.File(modelName, 'w') as m:
        metadata = model['metadata']
        activationScript = metadata['activationScript']
        mgrp = m.create_group('metadata')
        mgrp.create_dataset('activationScript', data=activationScript, shape=len(activationScript), dtype=h5.string_dtype())
        keys = list(model.keys())
        keys.remove('metadata')
        for key in keys:
            m.create_group(str(key))
            wgrp = m[str(key)]
            wgrp['weights'] = model[key]['weights']
            wgrp['biases'] = model[key]['biases']
        print(f'Model saved as {modelName}')
    return

#Applies our network against a testing dataset, it shouldn't include any identical specimens from our training set
def testNetwork(model, testingData, actScript=None):
    #If the variable testingData comes in as a file name string, readMnist will activate. If it was already previously converted into an array, it will pass
    if type(testingData) == str:
        testingData = readMnist(testingData)
    else:
        pass
    #Same deal as the testingData, if model is passed as a string, it will be converted. If not, it will be reassigned to wb and continues
    if type(model) == str:
        wb = readLayers(model)
    elif type(model) == dict:
        wb = model
    if actScript == None:
        actScript = actHandler(wb['metadata']['activationScript'])[0]
    right, wrong = 0, 0
    loss = 0.0
    for data in testingData:
        target = int(data[0])
        outputNum, prediction, outputLayer = runNetwork(filename=wb,data=data[1:], actScript=actScript)
        loss += calcLoss(target, np.array(outputLayer))
        if outputNum == target:
             right +=1
        else:
             wrong +=1
             #print(f'Target Num: {int(data[0])} Predicted Num: {outputNum}, Prediction: {prediction}, Loss: {calcLoss(int(data[0]), outputLayer)}', '\n', outputLayer, '\n')
             

    print(f'Target Num: {int(data[0])} Predicted Num: {outputNum}, Prediction: {prediction}, Loss: {calcLoss(int(data[0]), outputLayer)}', '\n', outputLayer, '\n')
    print(f'Right: {right} \nWrong: {wrong} \nPercentage: {100*right/len(testingData)}% \nLoss: {loss/len(testingData)}')
    return

if __name__ == '__main__':
    np.set_printoptions(precision=8, suppress=True, linewidth=200)
    #buildNetwork(784, (64,), 10, ['relu','relu','sigmoid'], 'test.hdf5')
    #untrained = readLayers('default.hdf5')[1]['weights']
    #trained = readLayers('layer.hdf5')[1]['weights']
    #diff = trained-untrained
    #parseImages(wb[1]['weights'])
    #saveImage(parseImages(trained), 'default.png')
    #saveImage(parseImages(untrained),'duntrained.png')
    #saveImage(parseImages(diff), 'diff.png')
