# TF - IDF（termfrequency–inversedocument frequency）
# 是一种用于资讯检索与资讯探勘的常用加权技术。
# TF - IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
# 字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
# TF - IDF加权的各种形式常被搜寻引擎应用，作为文件与用户查询之间相关程度的度量或评级。

import numpy as np
import pandas as pd

# 定义数据和预处理
docA = "The cat cat sat on my bed"
docB = "The dog dog sat on my knees"

bowA = docA.split(" ")
bowB = docB.split(" ")

#构建词库
wordSet = set(bowA).union(set(bowB))

# 进行词数的统计(wordCount)
wordDictA = dict.fromkeys(wordSet,0)
wordDictB = dict.fromkeys(wordSet,0)

# 遍历文档 统计词数
for word in bowA:
    wordDictA[word] += 1
for word in bowB:
    wordDictB[word] += 1

df = pd.DataFrame([wordDictA,wordDictB])
print(df)

# 计算词频 TF = 词i出现的次数 / 文档中的所有词数
def computeTF(wordDict,bow):
    # 使用一个dict 存储tf
    tfDict = {}
    nbowCount = len(bow)

    for word,count in wordDict.items():
        tfDict[word] = count/nbowCount
    return tfDict

tfA = computeTF(wordDictA,bowA)
tfB = computeTF(wordDictB,bowB)
print(tfA)
print(tfB)

#计算逆文档频率 log( 总文档数+1  /  有词语i的文档+1 )  值与有词语i的文档负相关
def computeIDF( wordDictList ):
    # 用一个字典对象保存idf结果 每个词作为key 值为对应的idf值
    idfDict = dict.fromkeys(wordDictList[0],0)
    N = len(wordDictList)
    import math
    #遍历所有文档
    for wordDict in  wordDictList:
        #遍历字典中的每个词汇
        for word,count in wordDict.items():
            if count >0 :
                idfDict[word] += 1

    #已经得到所有词汇i对应的Ni 将其替换为idf值
    for word,Ni in idfDict.items():
        idfDict[word] = math.log10((N + 1)/(Ni + 1))
    return idfDict

idfs = computeIDF([wordDictA,wordDictB])
print(idfs)

#计算TF-IDF
def computeTFIDF(tf,idfs):
    tfidf = {}
    for word,tfval in tf.items():
        tfidf[word] = tfval * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA,idfs)
tfidfB = computeTFIDF(tfB,idfs)

tfidfRes = pd.DataFrame([tfidfA,tfidfB])

print(tfidfRes)