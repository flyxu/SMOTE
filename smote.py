# coding=utf-8
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import process
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from unbalanced_dataset import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
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
    print '原始数据集的大小:',data.shape

    #split the data with 70% and 30%
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data[:,0:-1], data[:,-1], test_size=0.3, random_state=0)
    print '划分数据集后两个数据集的大小:',x_train.shape,x_test.shape

    N=process.get_ratio(y_train)
    train_x=np.c_[x_train,y_train]
    majority_samples,minority_samples=process.get_classsample(train_x)
    print '不平衡率：',N
    print '70%数据集中多数类和少数类的大小：',majority_samples.shape,minority_samples.shape
    s=Smote(samples=minority_samples[:,0:-1],N=N,k=5)
    synthetic= s.over_sampling()
    print '合成的少数类样本的大小:',synthetic.shape
    #将合成的少数类样本和原来70%数据集合并
    synthetic=np.c_[synthetic,np.array([1]*synthetic.shape[0])]
    train_data=np.vstack((synthetic,train_x))
    # train_minority=np.vstack((synthetic,minority_samples))
    # train_data=np.vstack((train_minority,majority_samples))
    # train_label = np.array([1] * train_minority.shape[0] + [0] *majority_samples.shape[0])
    print '最终用于训练的数据集的大小:',train_data.shape
    print '------------------------------'
    verbose = False
    ratio = float(np.count_nonzero(y_train == 0)) / float(np.count_nonzero(y_train == 1))
    smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
    smox, smoy = smote.fit_transform(x_train, y_train)
    print '用python模块smote处理后训练集的大小',smox.shape,smoy.shape
    print '------------------------------'

    print 'origin data各个评价指标'
    clf=svm.SVC()
    scores = cross_validation.cross_val_score(clf, data[:,0:-1], data[:,-1], cv = 10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accruacy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    print accruacy, precision, recall, f1_score
    confusion = confusion_matrix(y_test, y_pred)
    print confusion
    print '------------------------------'

    print 'smote data各个评价指标'
    clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
    clf1.fit(smox, smoy)
    y_pred1=clf1.predict(x_test)
    # clf1=svm.SVC()
    # scores1 = cross_validation.cross_val_score(clf1, smox, smoy, cv = 10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
    # clf1.fit(smox,smoy)
    # print smoy
    # y_pred1=clf1.predict(x_test)
    print y_pred1
    print y_test
    accruacy = metrics.accuracy_score(y_test, y_pred1)
    precision = metrics.precision_score(y_test, y_pred1)
    recall = metrics.recall_score(y_test, y_pred1)
    f1_score = metrics.f1_score(y_test, y_pred1)
    print accruacy, precision, recall, f1_score
    confusion = confusion_matrix(y_test, y_pred1)
    print confusion
    print '------------------------------'

    print 'mysmote data各个指标'
    clf2=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
    scores2=cross_validation.cross_val_score(clf2,train_data[:,0:-1],train_data[:,-1],cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
    clf2.fit(train_data[:,0:-1], train_data[:,-1])
    print train_data[:,-1]
    y_pred2 = clf2.predict(x_test)
    print y_pred2
    print y_test
    accruacy = metrics.accuracy_score(y_test, y_pred2)
    precision = metrics.precision_score(y_test, y_pred2)
    recall = metrics.recall_score(y_test, y_pred2)
    f1_score = metrics.f1_score(y_test, y_pred2)
    print accruacy, precision, recall, f1_score
    confusion = confusion_matrix(y_test, y_pred2)
    print confusion
    print '-------------------------------'

