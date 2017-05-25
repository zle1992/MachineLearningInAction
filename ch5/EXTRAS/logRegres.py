# -*- coding: utf-8 -*-
'''
Created on 14/5/2017
Logistic Regression
@author: zle1992
'''
import numpy as np
def loadDataSet():
    dataMat,labelMat = [],[]
    with open(filename,"r") as  fr:  #open file
        for line in fr.readlines():
            lineArr = line.split() #split each line
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  #创建2维list
            labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMat,labelMat):
    dataMatrix = np.mat(dataMat)  #translate list to matrix
    labelMatrix = np.mat(labelMat).transpose() #转置
    m,n = np.shape(dataMatrix) #100 rows  3 coulums
    alpha = 0.001 #步长 or 学习率
    maxCyclse = 500
    weight = np.ones((n,1)) #初始值随机更好吧
    #weight = np.random.rand(n,1)
    for k in range(maxCyclse):
        h = sigmoid(dataMatrix * weight) # h 是向量
        error = (labelMatrix - h)  #error 向量
        weight = weight + alpha * dataMatrix.transpose() *error  #更新
     #   print(k,"  ",weight)
    return weight

def plotfit(wei):
    import matplotlib.pyplot as plt
    weight = np.array(wei) #???????? #return array
    dataMat ,labelMat = loadDataSet() 
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]  #row
    fig = plt.figure()   #plot
    ax = fig.add_subplot(111)
    ax.scatter(dataArr[:,1],dataArr[:,2],s =50, c = np.array(labelMat)+5) #散点图 #参考KNN 的画图
    x = np.arange(-3.0,3.0,0.1)   #画拟合图像
    y = (-weight[0] - weight[1] *x ) / weight[2]
    ax.plot(x,y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def stocGradAscent0(dataMatrix,labelMatrix):
    m,n = np.shape(dataMatrix)
    alpha = 0.1
    weight = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix * weight))
        error = labelMatrix[i] - h
        weight = weight + alpha * error * dataMatrix[i]
    return weight
def stocGradAscent1(dataMat,labelMat,numIter = 150):
    dataMatrix = np.mat(dataMat)  #translate list to matrix
    labelMatrix = np.mat(labelMat).transpose() #转置
    m,n = np.shape(dataMat)
    alpha = 0.1
    weight = np.ones(n) #float 
    #weight = np.random.rand(n)
    for j in range(numIter):
        dataIndex = list(range(m)) #range 没有del 这个函数　　所以转成list  del 见本函数倒数第二行
        for i in range(m):
            alpha = 4/(1.0 +j + i) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex))) #random.uniform(0,5) 生成0-5之间的随机数
            #生成随机的样本来更新权重。
            h = sigmoid(sum(dataMat[randIndex] * weight))
            error = labelMat[randIndex] - h
            weight = weight + alpha * error * np.array(dataMat[randIndex])
            del(dataIndex[randIndex]) #从随机list中删除这个
    return weight
if __name__ == '__main__':
    filename = "testSet.txt" 
    dataMat,labelMat = loadDataSet()
    #print(dataMat,"\n",labelMat)
    #weight = gradAscent(dataMat,labelMat)
    weight = stocGradAscent1(dataMat,labelMat)
    print(weight)
    plotfit(weight)






