import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import process
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
class Smote:
    def __init__(self, samples, N=100, k=5):
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0
        # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        # if self.N<100:
        #     N=100
        #     pass
        # if self.N%100!=0:
        #     raise ValueError('N must be <100 or multiple of 100')
        if not isinstance(self.N, int):
            raise ValueError('N must be integer')
        N = self.N
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print 'neighbors', neighbors
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance=False)[0]
            # print nnarray
            self._populate(N, i, nnarray)
        return self.synthetic

    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i] + gap * dif
            self.newindex += 1

    # return sorted index
    def indexarray(self, intx, dataset, k):
        datasetsize = dataset.shape(0)
        diff = np.tile(intx, (datasetsize, 1)) - dataset
        sqdiff = diff ** 2
        sqdistance = sqdiff.sum(axis=1)
        distance = sqdistance ** 0.5
        sortedistindex = distance.argsort()
        return sortedistindex

    #def get_traindata(self,filename):

if __name__ == '__main__':
    filename='./cmv.csv'
    data=process.loadDataSet(filename)
    print 'the shape of origin data:',data.shape
    #print data[:,-1]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data[:,0:-1], data[:,-1], test_size=0.3, random_state=0)
    print x_train.shape,x_test.shape
    N=process.get_ratio(y_train)
    train_x=np.c_[x_train,y_train]
    majority_samples,minority_samples=process.get_classsample(train_x)
    print N
    print minority_samples.shape
    s=Smote(samples=minority_samples,N=N,k=5)
    synthetic= s.over_sampling()
    print synthetic.shape
    train_minority=np.vstack((synthetic,minority_samples))
    train_data=np.vstack((train_minority,majority_samples))
    train_label = np.array([1] * train_minority.shape[0] + [0] *majority_samples.shape[0])
    print 'the shape of train_data:',train_data.shape
    print '------------------------------'
    print 'the accuracy of origin data'
    clf=svm.SVC()
    scores = cross_validation.cross_val_score(clf, data[:,0:-1], data[:,-1], cv = 10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print '------------------------------'
    print 'the accuracy of smote data'
    clf2=svm.SVC(C=1000)
    clf2.fit(train_data,train_label)
    score = clf2.score(x_test,y_test)
    print score
    print '------------------------------'
    y_pred = clf2.predict(x_test)
    print y_pred
    print y_test
    accruacy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    print accruacy, precision, recall, f1_score
    confusion = confusion_matrix(y_test, y_pred)
    print confusion
    # s = Smote()
    # data = s.loadDataSet('./cmv.csv')
    # s.count()
    # gen = s.over_sampling()
    # result = np.vstack((gen, a))
    # print result

    # print s.standard(a)
    # print s.loadDataSet('./cmv.csv')
    # print s.count('./cmv.csv')
    # print s.get_minority('./cmv.csv')
