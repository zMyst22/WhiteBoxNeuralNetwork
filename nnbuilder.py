import h5py
import numpy as np

def createNetwork(inputSize:int, hiddenLayers:tuple, outputSize:int, filename='default', weightRange=[-1,1], biasRange=[0]):
    layers = [inputSize]+list(hiddenLayers)+[outputSize]
    if len(biasRange) == 1:
        biasMin = biasRange[0]
        biasMax = biasRange[0]
    else:
        biasMin = biasRange[0]
        biasMax = biasRange[1]

    with h5py.File(f'{filename}.hdf5', 'w') as nn:
        wgrp = nn.create_group('weights')
        bgrp = nn.create_group('biases')
        for i in range(len(layers)-1):
            w = f'w{i+1}'
            b = f'b{i+1}'
            weight = np.random.uniform(weightRange[0],weightRange[1],(layers[i+1],layers[i]))
            bias = np.random.uniform(biasMin, biasMax,(layers[i+1],))
            dset = wgrp.create_dataset(w, data=weight)
            dset = bgrp.create_dataset(b, data=bias)
        print(f'File saved as: {filename}.hdf5')
        print(f'Network has {inputSize} input neurons, {len(hiddenLayers)} hidden layers, and {outputSize} output neurons.')
        print(f'The initial weight range is between {weightRange[0]} and {weightRange[1]}.')
        if len(biasRange)==1:
            print(f'Biases have been initialized at {biasRange[0]}.')
        else:
            print(f'Biases have been randomly initialized between {biasRange[0]} and {biasRange[1]}')

def readNetwork(filename:str):
    with h5py.File(f'{filename}.hdf5', 'r') as nn:
        weights = nn['weights'].keys()
        biases = nn['biases'].keys()
        for weight, bias in zip(weights, biases):
            print(weight, bias)

#createNetwork(5,(3,8,5),4)
readNetwork('default')