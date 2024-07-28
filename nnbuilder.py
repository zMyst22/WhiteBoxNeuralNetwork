import h5py
import numpy as np

def createNetwork(inputSize:int, hiddenLayers:tuple, outputSize:int, filename='default', weightRange=[0,1], biasRange=[0]):
    layers = [inputSize]+list(hiddenLayers)+[outputSize]
    
    if len(biasRange) == 1:
        biasMin, biasMax = biasRange[0], biasRange[0]
    else:
        biasMin, biasMax = biasRange

    with h5py.File(f'{filename}.hdf5', 'w') as nn:
        wgrp = nn.create_group('weights')
        bgrp = nn.create_group('biases')
        for i in range(len(layers)-1):
            w = f'w{i+1}'
            b = f'b{i+1}'
            weight = np.random.uniform(weightRange[0],weightRange[1],(layers[i+1],layers[i]))
            bias = np.random.uniform(biasMin, biasMax,(layers[i+1],)).round(2)
            wdset = wgrp.create_dataset(w, data=weight)
            bdset = bgrp.create_dataset(b, data=bias)
        print(f'File saved as: {filename}.hdf5')
        print(f'Network has {inputSize} input neuron(s), {len(hiddenLayers)} hidden layer(s), and {outputSize} output neuron(s).')
        print(f'The initial weight range is between {weightRange[0]} and {weightRange[1]}.')
        if len(biasRange)==1:
            print(f'Biases have been initialized at {biasRange[0]}.')
        else:
            print(f'Biases have been randomly initialized between {biasRange[0]} and {biasRange[1]}')

if __name__ == '__main__':
    createNetwork(5, (4,3),2)
