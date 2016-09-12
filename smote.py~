import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
class Smote:
    def __init__(self,samples,N=100,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))
    def loadDataSet(self,filename):
        data=pd.read_csv(filename)
        return data.values
    def count(self,filename):
        data=self.loadDataSet(filename)
        maj_count=np.count_nonzero(data[:,-1]=='N')
        min_count=np.count_nonzero(data[:,-1]=='Y')
        return 'min_cout:',min_count,'maj_count:',maj_count
    def get_minority(self,filename):
        data=self.loadDataSet(filename)
        majority=data[data[:,-1]=='N']
        minority=data[data[:,-1]=='Y']
        return minority[:,0:-1]
    #Standardization
    def standard(self,dataset):
        stdval=dataset.std(0)
        meanval=dataset.mean(0)
        normdataset=np.zeros(np.shape(dataset))
        m=dataset.shape[0]
        normdataset=dataset-np.tile(meanval,(m,1))
        normdataset=normdataset/np.tile(stdval,(m,1))
        return normdataset
    def over_sampling(self):
        if self.N<100:
            N=100
            pass
        if self.N%100!=0:
            raise ValueError('N must be <100 or multiple of 100')
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print 'neighbors',neighbors
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic
    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

if __name__=='__main__':

    a = np.array([[1, 2, 3], [3, 5, 6], [2, 3, 1]])
    s = Smote(a, N=100)
    gen = s.over_sampling()
    result = np.vstack((gen, a))
    print s.standard(a)
    print s.loadDataSet('./cmv.csv')
    #print s.count('./cmv.csv')
    #print s.get_minority('./cmv.csv')
