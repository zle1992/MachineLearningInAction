
from math import log
import operator
import pickle
import treePlotter
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels
import math

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #Data 的大小N,N行
    labelCount = {}#字典存储 不同类别的个数
    for featVec in dataSet :
        currentLabel = featVec[-1] #每行的最后一个是类别
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1  #原书缩进错误！！！
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/numEntries
        shannonEnt -= prob *math.log(prob,2) #熵最后外面有个求和符号 ！！！
    return shannonEnt
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #去掉axis 这一列
            reducedFeatVec = featVec[:axis] 
            reducedFeatVec.extend(featVec[axis+1: ])
            retDataSet.append(reducedFeatVec)
    return retDataSet  
def chooseBestFeatureTopSplit(dataSet):
    #列数 = len(dataset[0])
    #行数 = len(dataset)
    numFeatures = len(dataSet[0]) -1 #最后一列是标签  
    baseEntropy = calcShannonEnt(dataSet) #所有数据的信息熵
    bestInfoGainn = 0.0
    bestFeature = -1
    for i in range(numFeatures):#遍历不同的属性
        featList = [example[i] for example in dataSet] #取出每一列
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:#在第i个属性里，遍历第i个属性所有不同的属性值
            subDataSet = splitDataSet(dataSet,i,value) #划分数据 
            prob = len(subDataSet)/float(len(dataSet))  #len([[]]) 行数
            newEntropy += prob *calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGainn):
            bestInfoGainn = infoGain
            bestFeature = i
    return bestFeature
def majorityCnt(classList):
    classCount ={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
  
    classCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #与python2 不同！！！！！！python3 的字典items 就是迭代对象
    return classCount[0][0] #返回的是字典第一个元素的key 即 类别标签
def createTree(dataSet,labels):
    #mytree 是一个字典，key 是属性值，val 是类别或者是另一个字典，
    #如果val 是类标签，则该子节点就是叶子节点
    #如果val是另一个数据字典，则该节点是一个判断节点
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0])==1: #完全划分
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureTopSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat]) #
    featValues = [example[bestFeat] for example in dataSet] # 某属性的所有取值
    uniqueVals = set(featValues)
    for value in uniqueVals :
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
                    
    
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) #将标签转化成索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:  
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else : classLabel = secondDict[key]#到达叶子节点，返回标签
    return classLabel

def storeTree(inputTree,filename):

    with open(filename,'wb') as f:
        pickle.dump(inputTree,f)
    
def grabTree(filename):
    with open(filename,'rb') as f:
        t = pickle.load(f)
    return t
    
if __name__ == '__main__':
    dataSet,labels = createDataSet()
    tree = createTree(dataSet,labels)
    storeTree(tree,"tree.model")
    tree = grabTree("tree.model")
    treePlotter.createPlot(tree)

    #读取txt文件，预测隐形眼镜的类型
    with open('lenses.txt') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]    
    lensesLabels = ['age','prescript','astigmastic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    treePlotter.createPlot(lensesTree)
