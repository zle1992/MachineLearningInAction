#coding: utf-8  
import numpy as np 
import operator
import matplotlib
from numpy import *
import matplotlib.pyplot as plt 
import os 

def createDataSet():
    group = np.array([
    [1.0,1.1],
    [1.0,1.0],
    [0,0],
    [0,0.1] ])
    labels = ['A','A','B','B']
    return group ,labels
def classify0(intX,dataX,labels,k):
    dataSize = dataX.shape[0] #行数，
    diffMat = np.tile(intX,(dataSize,1)) - dataX #intX　复制４行，形成矩阵，并计算距离差
    sqDiffMat = diffMat * diffMat  # 等价于 diffMat ** 2
    sqDistence = sqDiffMat.sum(axis = 1) #按行相加
    distence = sqDistence ** 0.5 # 开根号 ，　distence是array
    sortedDistenceIndicies = distence.argsort() # 返回排序后的下标
    classCount = {} # 字典　key is label , val is count  
    for i in range(k):
        voteIlabel = labels[sortedDistenceIndicies[i]] # 排名第i的label
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #   .get(voteIlabel,0)   if no exit return 0
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True) 
    # 根据iteritems 的第１个元素排序　即字典中的val ,第０个　是key  #reverse定义为True时将按降序排列
    #!!!!! 与py2 的差别！！ python3 字典的item 就是迭代对象
    return sortedClassCount[0][0]  # 返回排序后的字典的前一个（从０开始）
def file2matrix(filename):
    with open(filename,mode = "r") as fr:
        arrayOLines = fr.readlines()
        numberOfLine = len(arrayOLines)
        returnMat = np.zeros((numberOfLine,3))
        labels = []
        index = 0
        for line in arrayOLines:
            listFromLine = line.split("\t")
            returnMat[index,:] = listFromLine[0:3]
            labels.append(int(listFromLine[-1]))  #-1 倒数第一个
            index = index + 1
        return returnMat,labels
def autoNorm(dataX):
    #  归一化公式 newVal = (oldval-min)/(max - min)
    minVals = dataX.min(axis=0)#返回的是最每一列中的最小的元素 min(1)#返回的是最每一行中的最小的元素
    maxVals = dataX.max(0)#返回的是最每一列中的最大的元素
    ranges = maxVals - minVals #
    rows = dataX.shape[0]
    newVal = dataX - tile(minVals,(rows,1)) #(oldval-min)
    ##   tile(minVals,(rows,1))复制rows行 ，列数为minvals列数的一倍
    newVal = newVal/tile(ranges,(rows,1)) #(oldval-min)/(max - min)
    return newVal,ranges,minVals
def datingClassTest():
    hoRatio = 0.1 # 10% of data as  test
    #读入数据
    filename = "datingTestSet2.txt"
    dataX ,labels = file2matrix(filename)
    #归一化
    normMat,ranges,minVals = autoNorm(dataX)

    m = dataX.shape[0] #numbers of rows
    numTestVecs = int(m * hoRatio) #numbers of test 
    errorcount = 0 #initialize number of errors 
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],labels[numTestVecs:m],5)#前10%作为测试数据
    #   print("the classifier predict %d, the real answer is :%d" %((classifierResult),labels[i]))
        if(classifierResult != labels[i]):
            errorcount = errorcount + 1.0
    print("error rate :%f" %((errorcount)/(numTestVecs)))

def classifyPerson():
    resultList = ["第一类","第二类","第三类"] #output lables
    
    percentTats = float(input("玩游戏消耗的时间"))###########!!!!!!! pyhton3 为input#####!!!!!!!!!!!!!!!!!
    ffilm = float(input("每年获得的飞行里程数"))###########!!!!!!! pyhton2 为raw_input#####!!!!!!!!!!!!!!!!!
    iceCream = float(input("每周消费冰淇淋"))
        #读入数据
    filename = "datingTestSet2.txt"
    dataX ,labels = file2matrix(filename)
    #归一化
    normMat,ranges,minVals = autoNorm(dataX)

    test_list = np.array([percentTats,ffilm,iceCream])
    classifierResult = classify0(test_list,dataX,labels,3)
    print("你喜欢的类别:" + resultList[classifierResult])
def img2vector(filename):
    returnVect = np.zeros((1,1024)) #initilize the vec
    with open (filename,mode = "r") as fr:   #########!!!!!!!!与python2 不同   ！！！！！！！！
        lineStr = fr.readlines()             #########!!!!!!!!与python2 不同   ！！！！！！！！
        for i in range(32):                  #########!!!!!!!!与python2 不同   ！！！！！！！！
            for j in range(32):                 #########!!!!!!!!与python2 不同   ！！！！！！！！
                returnVect[0,i*32+j] = lineStr[i][j]        #########!!!!!!!!与python2 不同   ！！！！！！！！
    return returnVect
def handwritingClassTest():
    hwlables = []
    trainingFileList = os.listdir("digits\\trainingDigits")######!!!!!!!!与python2 不同   ！py3 need import os 
    m = len(trainingFileList) # number of trianfiles
    trainMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]   #循环操作 把每一个txt文件转换为矩阵
        fileStr = fileNameStr.split(".")[0]  # 把文件名与后缀名分开，提取文件名
        classNumStr = fileStr.split("_")[0]  #提取文件名中的标签
        hwlables.append(classNumStr)  #把类别标签添加到类别List
        trainMat[i,:] = img2vector("digits\\trainingDigits\\%s" %fileNameStr)

    testFileList = os.listdir("digits\\testDigits")  #处理测试文件 
    mtest = len(testFileList)
    errorcount = 0.0 
    for i in range(mtest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = fileStr.split("_")[0]
        vectorUderTest = img2vector("digits\\testDigits\\%s" %fileNameStr)
        classifierResult = classify0(vectorUderTest,trainMat,hwlables,3)
        if(classifierResult != classNumStr) :
            errorcount += 1.0
    print("errorcount:%d" %errorcount)
    print("error rate :%f" %float((errorcount/mtest)))


def simpletest():  # 测试createDataSet 和classify0 这2个函数
    intX = [0.0,0.0]
    k = 3
    dataX,labels = createDataSet() 
    a = classify0(intX,dataX,labels,k)
    print("dataX:" ,dataX)
    print("labels:",labels)
    print("predict:" ,a)
def plot(): #画datingTestSet2.txt这个数据的图像
    k = 3
    #读入数据
    filename = "datingTestSet2.txt"
    dataX,labels = file2matrix(filename)
    #归一化
    normMat,ranges,minVals = autoNorm(dataX)
    intX = normMat[:1,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataX[:,0],dataX[:,1],c = 15 *np.array(labels),s = 15*np.array(labels)) # c 是颜色序列！！！！ s 是大小
    ax = fig.add_subplot(121)
    ax.scatter(dataX[:,0],dataX[:,2],c = 15 *np.array(labels),s = 15*np.array(labels)) # c 是颜色序列！！！！ s 是大小
    ax = fig.add_subplot(131)
    ax.scatter(dataX[:,1],dataX[:,2],c = 15 *np.array(labels),s = 15*np.array(labels)) # c 是颜色序列！！！！ s 是大小
    #print(labels)
    plt.show()
if __name__ == '__main__':
    simpletest()   # 测试createDataSet 和classify0 这2个函数
    #plot()   #画datingTestSet2.txt这个数据的图像
    #handwritingClassTest()  #手写识别程序
    #datingClassTest()  #约会 识别程序
    #classifyPerson()  #从命令行读入数据
   
   


