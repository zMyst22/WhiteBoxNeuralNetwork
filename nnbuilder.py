import h5py
import numpy as np

def createNetwork(inputSize:int, hiddenLayers:tuple, outputSize:int, filename='default.hdf5', weightRange=[-1,1], biasRange=[-1,1], precision=2):
    if hiddenLayers == None:
        hiddenLayers = []
        layers =[inputSize]+[outputSize]
    else:
        layers = [inputSize]+list(hiddenLayers)+[outputSize]
    
    if len(biasRange) == 1:
        biasMin, biasMax = biasRange[0], biasRange[0]
    else:
        biasMin, biasMax = biasRange

    with h5py.File(filename, 'w') as nn:
        for layer in range(len(layers)-1):
            weights = np.random.uniform(weightRange[0],weightRange[1],(layers[layer+1],layers[layer])).round(precision)
            biases = np.random.uniform(biasMin, biasMax,(layers[layer+1],)).round(precision)
            lgrp = nn.create_group(str(layer+1))
            wgrp = lgrp.create_dataset('weights', data=weights)
            bgrp = lgrp.create_dataset('biases', data=biases)
        print(f'File saved as: {filename}')
        print(f'Network has {inputSize} input neuron(s), {len(hiddenLayers)} hidden layer(s), and {outputSize} output neuron(s).')
        print(f'The initial weight range is between {weightRange[0]} and {weightRange[1]}.')
        if len(biasRange)==1:
            print(f'Biases have been initialized at {biasRange[0]}.')
        else:
            print(f'Biases have been randomly initialized between {biasRange[0]} and {biasRange[1]}')

if __name__ == '__main__':
    createNetwork(784, None ,10, filename='test.hdf5')
