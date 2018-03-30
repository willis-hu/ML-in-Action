from numpy import *
import operator     #运算符模块
import matplotlib
import matplotlib.pyplot as plt

class kNN(object):

    def createDataSet(self):
        group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
        labels = ['A','A','B','B']
        return group,labels

    def clasify0(self,inX,dataSet,labels,k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inX,(dataSetSize,1)) - dataSet   #将inX变换为何dataSet一样大小，相减
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5        #计算inX与每个样本的距离
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i  in range(k):
            voteIlable = labels[sortedDistIndicies[i]]
            classCount[voteIlable] = classCount.get(voteIlable,0)+1     #取距离最小的k个样本，labels计数+1
        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
        # operator.itemgetter(n)返回一个函数，该函数取对象的第n域的值。
        # 返回距离计数最大的类
        return sortedClassCount[0][0]

    def file2matrix(self,filename):
        fr = open(filename)
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines,3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat,classLabelVector

    def autoNorm(self,dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals-minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals,(m,1))
        normDataSet = normDataSet/tile(ranges,(m,1))
        return normDataSet,ranges,minVals

#   该方法传入参数为k
    def datingClassTest(self,k):
        hoRatio = 0.10
        datingDataMat,datingLabels = self.file2matrix('datingTestSet2.txt')
        normMat,ranges,minVals = self.autoNorm(datingDataMat)
        m = normMat.shape[0]
        numTestVecs = int(m*hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = self.clasify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
            print("the classifier came back with : %d,the real answer is : %d"%(classifierResult,datingLabels[i]))
            if(classifierResult != datingLabels[i]):errorCount += 1.0
        print("the total error rate is : %f"%(errorCount/float(numTestVecs)))

if __name__ == '__main__':
    a = [0,0]
    t = kNN()
    t.datingClassTest(3)


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.show()