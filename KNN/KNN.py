import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    love_dictionary = {'largeDoses': 3,
                       'smallDoses': 2,
                       'didntLike': 1}
    with open(filename, 'r') as fr:
        arrayOfLines = fr.readlines()
        numberOfLines = len(arrayOfLines)
        returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return
        classLabelVector = []  # prepare labels return
        index = 0
        for rawline in arrayOfLines:
            line = rawline.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            if (listFromLine[-1].isdigit()):
                classLabelVector.append(int(listFromLine[-1]))
            else:
                classLabelVector.append(love_dictionary.get(listFromLine[-1]))
            index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10  #hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        #print("The calssifier came back with %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
            print("The calssifier came back with %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
    print("The total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses' ]
    percetTats = float(input("percetage of time spent play video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percetTats, iceCream,])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    errorCount = 0.0
    testFileList = os.listdir('testDigits')  # iterate through the test set
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        if classifierResult != classNumStr:
            errorCount += 1.0
            print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
    print("\n The total number of error is: %d" % errorCount)
    print("\n The total error rate is: %f" % (errorCount/float(mTest)))


def plot():
    datingDataMat, datingLables = file2matrix('datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 2],
               15.0*np.array(datingLables), 15*np.array(datingLables))
    plt.show()


def main():
    #group, labels = createDataSet()
    #res = classify0([0, 0], group, labels, 3)
    #print(res)

    #plot()

    #datingDataMat, datingLables = file2matrix('datingTestSet.txt')
    #normMat, ranges, minVals = autoNorm(datingDataMat)
    #print(ranges)
    #print(minVals)

    #datingClassTest()

    #classifyPerson()

    handwritingClassTest()


if __name__ == '__main__':
    main()
