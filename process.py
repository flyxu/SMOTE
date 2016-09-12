import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
#load data
def loadDataSet(filename):
    data = pd.read_csv(filename)
    data['Defective']=data['Defective'].map({'Y':1, 'N':0})
    return data.values
#the ratio of majority and minority
def get_ratio(label):
    #data = loadDataSet(filename)
    maj_count = np.count_nonzero(label == 0)
    min_count = np.count_nonzero(label == 1)
    print 'maj_count:', maj_count, 'min_count:', min_count
    return maj_count / min_count
#get the minority and majority samples and drop the labels
def get_classsample(data):
    #data = loadDataSet(filename)
    majority = data[data[:, -1] == 0]
    minority = data[data[:, -1] == 1]
    return majority[:,0:-1],minority[:, 0:-1]
#Standardization
def standard(self,dataset):
    stdval=dataset.std(0)
    meanval=dataset.mean(0)
    normdataset=np.zeros(np.shape(dataset))
    m=dataset.shape[0]
    normdataset=dataset-np.tile(meanval,(m,1))
    normdataset=normdataset/np.tile(stdval,(m,1))
    return normdataset
