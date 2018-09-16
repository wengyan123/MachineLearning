# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   # 分别是X0，X1，X2
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)              # 100 * 3的矩阵
    labelMat = np.mat(classLabels).transpose()  # 100 * 1的列向量
    m, n = np.shape(dataMatrix)                 # 行数（样本数：100）、列数（特征数：3）
    alpha = 0.001                                # 目标移动的步长
    maxCycles = 500                             # 迭代次数
    weights = np.ones((n, 1))                   # 初始回归系数。3*1的列向量，每个系数初始化为1.0。weights为numpy.matrix型
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)         # 矩阵乘法：每个样本的特征值×系数，拔得出来的值作为sigmoid函数输入
        error = (labelMat - h)
        # https://blog.csdn.net/CharlieLincy/article/details/70767791?locationNum=11&fps=1
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()

    #weights = gradAscent(dataArr, labelMat)
    #print(weights)
    #plotBestFit(weights.getA())

    #weights = stocGradAscent0(np.array(dataArr), labelMat)
    #plotBestFit(weights)

    weights = stocGradAscent1(np.array(dataArr), labelMat)
    plotBestFit(weights)
