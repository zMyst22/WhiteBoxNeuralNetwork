import nnbuilder as nb
import nntrainer as nt
import nnrun as nr
import csv
import numpy as np
import h5py as h5
import time as t

if __name__ == '__main__':
    start = t.time()
    np.set_printoptions(precision=8, suppress=True, linewidth=200)
    with open('short_mnist_test.csv','r') as f:
        def setUp(row=np.ndarray):
                row = np.array(row, dtype=float)
                label = row[0]
                row/=255
                row[0] = label
                return row  
        
        reader = csv.reader(f)
        trainingData = list(map(setUp, reader))
        trainingData = np.array(trainingData).round(2).tolist()
    right = 0
    wrong = 0
    loss = 0.0
    wb = nr.readLayers('layer.hdf5')
    for data in trainingData:
        target = int(data[0])
        #Left to right
        outputNum, prediction, outputLayer = nr.runNetwork(filename=wb,data=data[1:], actScript=[nr.relu, nr.relu, nr.sigmoid])
        #if outputNum != int(data[0]):
        #    print(f'Target Num: {int(data[0])} Predicted Num: {outputNum}, Prediction: {prediction}, Loss: {nt.calcLoss(int(data[0]), outputLayer)}', '\n', outputLayer, '\n')


        loss += nt.calcLoss(target, np.array(outputLayer))
        if outputNum == target:
             right +=1
        else:
             wrong +=1
    
    print(f'Target Num: {int(data[0])} Predicted Num: {outputNum}, Prediction: {prediction}, Loss: {nt.calcLoss(int(data[0]), outputLayer)}', '\n', outputLayer, '\n')
    print(f'Right: {right} \nWrong: {wrong} \nPercentage: {100*right/len(trainingData)}% \nLoss: {loss/len(trainingData)}')
    end = t.time()
    print("Time: ", end-start)