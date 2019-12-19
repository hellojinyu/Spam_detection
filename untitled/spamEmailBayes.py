# encoding=utf-8
import jieba;
import os;
import re;


class spamEmailBayes:
    # 读取停用词文件，返回停用词表
    def getStopWords(self):
        stopWordsList = []
        for line in open("./data/stopWords.txt"):
            stopWordsList.append(line[:len(line) - 1])
        return stopWordsList;

    # 获得邮件经过分词和停用词过滤的分词列表
    def get_word_list(self, content, wordsList, stopWordsList):
        seg_list = list(jieba.cut(content))
        for i in seg_list:
            if i not in stopWordsList and i.strip() != '' and i != None:
                if i not in wordsList:
                    wordsList.append(i)

    # 用于形成初始的垃圾邮件词典和正常邮件词典，存放的是对应词的出现次数
    # 若列表中的词已在词典中，则次数加1，否则直接添加进去，次数赋为1
    def addToDict(self, wordsList, wordsDict):
        for item in wordsList:
            if item in wordsDict.keys():
                wordsDict[item] += 1
            else:
                wordsDict.setdefault(item, 1)

    # 计算正常邮件中的词频，构成词典hamDict，key是词，值是该词出现的频率或者频率的估计作为先验概率在后续使用
    def get_hamDict(self, hammFileList, stopWordsList, traindir, r):
        # 获取训练集中正常邮件的数量
        hamFilelen = len(hammFileList)
        wordsList = []
        hamDict = {}
        for fileName in hammFileList:
            wordsList.clear()
            for line in open(traindir + fileName):
                # 过滤掉非中文字符
                rule = re.compile(r"[^\u4e00-\u9fa5]")
                line = rule.sub("", line)
                # 将每封邮件出现的词保存在wordsList中
                email.get_word_list(line, wordsList, stopWordsList)
            # 将该邮件的词添加进词典hamDict
            email.addToDict(wordsList, hamDict)
        # 由词典中统计的词出现的次数，计算词出现的概率，形成真正的词典hamDict
        for word in hamDict.keys():
            # 统计这个词在正常邮件中的频率或频率的估计
            hamDict[word] = (hamDict[word] + r) / (hamFilelen + 2 * r)
        return hamDict

    # 计算垃圾邮件中的词频,构成词典spamDict，key是词，值是该词出现的频率
    def get_spamDict(self, spamFileList, stopWordsList, traindir, r):
        # 获取训练集中垃圾邮件的数量
        spamFilelen = len(spamFileList)
        wordsList = []
        spamDict = {}
        # 获得垃圾邮件中的词频
        for fileName in spamFileList:
            wordsList.clear()
            for line in open(traindir + fileName):
                # 过滤掉非中文字符
                rule = re.compile(r"[^\u4e00-\u9fa5]")
                line = rule.sub("", line)
                # 将每封邮件出现的词保存在wordsList中
                email.get_word_list(line, wordsList, stopWordsList)
            # 将该邮件的词添加进词典spamDict
            email.addToDict(wordsList, spamDict)
        # 由词典中统计的词出现的次数，计算词出现的概率，形成真正的词典spamDict
        for word in spamDict.keys():
            # 统计这个词在垃圾邮件中的频率或频率的估计
            spamDict[word] = (spamDict[word] + r) / (spamFilelen + 2 * r)
        return spamDict

    # 从testList中计算测试邮件中每个词的后验概率，来得到对分类影响最大的15个词（及后验概率最大的15个词）
    def getMaxTestWords(self, testList, spamDict, hamDict, ps, ps_condition, ph_condition):
        # ps为单词属于垃圾的先验概率，ph为属于正常的先验概率 ps_condition是条件概率
        ph = (1 - ps)
        wordProbList = {}
        # 根据spamDict和hamDict查找类条件概率，计算后验概率
        for word in testList:
            # 该词word在spamDict和hamDict中都有出现，直接查找类条件概率
            if word in spamDict.keys() and word in hamDict.keys():
                # 该文件中包含词个数
                ts = spamDict[word]
                th = hamDict[word]
                p_s = ts * ps / (ts * ps + th * ph)
                wordProbList.setdefault(word, p_s)
            # 该词word只在spamDict中出现，属于正常的类条件概率赋为0.01
            if word in spamDict.keys() and word not in hamDict.keys():
                ts = spamDict[word]
                th = ph_condition
                p_s = ts * ps / (ts * ps + th * ph)
                wordProbList.setdefault(word, p_s)
            # 该词word只在hamDict中出现，属于垃圾的类条件概率赋为0.01
            if word not in spamDict.keys() and word in hamDict.keys():
                ts = ps_condition
                th = hamDict[word]
                p_s = ts * ps / (ts * ps + th * ph)
                wordProbList.setdefault(word, p_s)
            # 测试邮件中的单词不在任何一个词典
            if word not in spamDict.keys() and word not in hamDict.keys():
                # 若该词不在脏词词典中，概率设为p
                ts = ps_condition
                th = ph_condition
                p_s = ts * ps / (ts * ps + th * ph)
                wordProbList.setdefault(word, p_s)
        # 逆序排序，从小到大，取后验概率最大的15个词
        sorted(wordProbList.items(), key=lambda d: d[1], reverse=True)[0:15]
        return wordProbList

    # 计算贝叶斯概率，并判断属于垃圾邮件还是正常邮件
    def calBayesProb(self, wordProbList, s):
        # s为根据wordProbList判断垃圾邮件和正常邮件的阈值
        prob1 = 1
        prob2 = 1
        for word, prob in wordProbList.items():
            prob1 *= prob
            prob2 *= (1 - prob)
        p = prob1 / (prob1 + prob2)
        if p > s:  # 大于阈值，判断属于垃圾邮件
            result = 1
        else:  # 否则，属于正常邮件
            result = 0
        return result

    # 对测试集中的邮件进行测试
    def testFile(self, stopWordsList, spamDict, hamDict, testspamFileList, testhamFileList, s, ps, ps_condition,
                 ph_condition, testResult,
                 testdir):
        # testList保存每封邮件中出现的词
        testList = []
        # 对真实的垃圾邮件进行测试
        for fileName in testspamFileList:
            testList.clear()
            for line in open(testdir + fileName):
                # 用正则表达式过滤非中文字符
                rule = re.compile(r"[^\u4e00-\u9fa5]")
                line = rule.sub("", line)
                # 根据读取结果line，进行分词和停用词过滤，添加进testList
                email.get_word_list(line, testList, stopWordsList)
            # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
            wordProbList = email.getMaxTestWords(testList, spamDict, hamDict, ps, ps_condition, ph_condition)
            # 对每封邮件得到的15个词计算贝叶斯概率
            result = email.calBayesProb(wordProbList, s)
            if result == 1:
                testResult.setdefault('1' + fileName, 1)
            else:
                testResult.setdefault('1' + fileName, 0)
        for fileName in testhamFileList:
            testList.clear()
            for line in open(testdir + fileName):
                # 用正则表达式过滤非中文字符
                rule = re.compile(r"[^\u4e00-\u9fa5]")
                line = rule.sub("", line)
                # 根据读取结果line，进行分词和停用词过滤，添加进testList
                email.get_word_list(line, testList, stopWordsList)
            # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
            wordProbList = email.getMaxTestWords(testList, spamDict, hamDict, ps, ps_condition, ph_condition)
            # 对每封邮件得到的15个词计算贝叶斯概率
            result = email.calBayesProb(wordProbList, s)
            if result == 1:
                testResult.setdefault('0' + fileName, 1)
            else:
                testResult.setdefault('0' + fileName, 0)
        return testResult


def test(traindir, testdir, r):
    testResult = {}
    S = 0  # 测试的垃圾邮件总数
    H = 0  # 测试的正常邮件总数
    SS = 0  # 统计spam判为spam的数量
    HS = 0  # 统计ham判为spam的数量
    SH = 0  # 统计spam判为ham的数量
    HH = 0  # 统计ham判为ham的数量
    trainhamFileList = []  # 获得训练集中正常邮件文件名称列表
    trainspamFileList = []  # 获取训练集中垃圾邮件文件名称列表
    testhamFileList = []  # 获取测试集中正常邮件文件名称列表
    testspamFileList = []  # 获取测试集中垃圾邮件文件名称列表
    trainFileList = os.listdir(traindir)
    for file in trainFileList:
        if file.split('_')[0] == "ham":  # 训练邮件是正常邮件
            trainhamFileList.append(file)  # 放入训练正常邮件集中
        else:  # 否则放入训练垃圾邮件集中
            trainspamFileList.append(file)
    testFileList = os.listdir(testdir)
    for file in testFileList:
        if file.split('_')[0] == "ham":  # 测试邮件是正常邮件
            testhamFileList.append(file)  # 放入测试正常邮件集中
        else:  # 否则放入测试垃圾邮件集中
            testspamFileList.append(file)
    s = 0.9  # s为阈值，若测试邮件最终概率大于s，则该邮件为垃圾邮件，否则为正常邮件，s=0.5时为后验概率最大化准则
    lenspam = len(trainspamFileList)  # 训练的垃圾邮件的个数
    lenham = len(trainhamFileList)  # 训练的正常邮件的个数
    ps_condition = r / (lenspam + 2 * r)  # 出现在垃圾邮件中的条件概率
    ph_condition = 1 - ps_condition  # 出现在正常邮件中的条件概率
    n = lenspam + lenham  # 训练的邮件的总个数
    ps = (lenspam + r) / (n + 2 * r)  # 垃圾邮件的先验概率
    # 获得停用词表，用于停用词过滤
    stopWordsList = email.getStopWords()
    # 获取正常邮件词典
    hamDict = email.get_hamDict(trainhamFileList, stopWordsList, traindir, r)
    # 获取垃圾邮件词典
    spamDict = email.get_spamDict(trainspamFileList, stopWordsList, traindir, r)
    # 调用测试函数，获取测试结果
    testResult = email.testFile(stopWordsList, spamDict, hamDict, testspamFileList, testhamFileList, s, ps,
                                ps_condition, ph_condition,
                                testResult, testdir)
    # 根据测试结果计算评价指标
    for key, catagory in testResult.items():
        if (int(key[0]) == 0 and catagory == 1):
            # 在此情况下是ham被判为spam，判断错误，则HS加1，H加1
            HS = HS + 1
            H = H + 1
        elif (int(key[0]) == 1 and catagory == 0):
            # 在此情况下是spam被判为ham，判断错误，则SH加1，S加1
            SH = SH + 1
            S = S + 1
        elif (int(key[0]) == 0 and catagory == 0):
            # 在此情况下是ham被判为ham，判断正确，则HH加1，H加1
            HH = HH + 1
            H = H + 1
        elif (int(key[0]) == 1 and catagory == 1):
            # 在此情况下是spam被判为spam，判断正确，则SS加1，S加1
            SS = SS + 1
            S = S + 1
    # 计算相关评价指标
    recall = (SS / (SS + SH)) * 100  # 召回率
    precision = (SS / (SS + HS)) * 100  # 精确率
    f1 = 100 * 2 * SS / (2 * SS + HS + SH)  # F1值
    return recall, precision, f1


if __name__ == '__main__':
    r = 1.0
    email = spamEmailBayes()  # email为类对象
    # testResult保存预测结果,key为真实类别（0代表正常邮件，1代表垃圾邮件）+文件名
    # 值为预测的邮件类别（0代表正常邮件，1代表垃圾邮件）
    traindir1 = "./data/email/ten_fold_cross/1/train/"
    testdir1 = "./data\email/ten_fold_cross/1/test/"
    recall1, precision1, f11 = test(traindir1, testdir1, r)
    print("序号1：")
    print("召回率(%)：", recall1)
    print("正确率(%)：", precision1)
    print("F1值(%)：", f11)
    traindir2 = "./data/email/ten_fold_cross/2/train/"
    testdir2 = "./data\email/ten_fold_cross/2/test/"
    recall2, precision2, f12 = test(traindir2, testdir2, r)
    print("序号2：")
    print("召回率(%)：", recall2)
    print("正确率(%)：", precision2)
    print("F1值(%)：", f12)
    traindir3 = "./data/email/ten_fold_cross/3/train/"
    testdir3 = "./data\email/ten_fold_cross/3/test/"
    recall3, precision3, f13 = test(traindir3, testdir3, r)
    print("序号3：")
    print("召回率(%)：", recall3)
    print("正确率(%)：", precision3)
    print("F1值(%)：", f13)
    traindir4 = "./data/email/ten_fold_cross/4/train/"
    testdir4 = "./data\email/ten_fold_cross/4/test/"
    recall4, precision4, f14 = test(traindir4, testdir4, r)
    print("序号4：")
    print("召回率(%)：", recall4)
    print("正确率(%)：", precision4)
    print("F1值(%)：", f14)
    traindir5 = "./data/email/ten_fold_cross/5/train/"
    testdir5 = "./data\email/ten_fold_cross/5/test/"
    recall5, precision5, f15 = test(traindir5, testdir5, r)
    print("序号5：")
    print("召回率(%)：", recall5)
    print("正确率(%)：", precision5)
    print("F1值(%)：", f15)
    traindir6 = "./data/email/ten_fold_cross/6/train/"
    testdir6 = "./data\email/ten_fold_cross/6/test/"
    recall6, precision6, f16 = test(traindir6, testdir6, r)
    print("序号6：")
    print("召回率(%)：", recall6)
    print("正确率(%)：", precision6)
    print("F1值(%)：", f16)
    traindir7 = "./data/email/ten_fold_cross/7/train/"
    testdir7 = "./data\email/ten_fold_cross/7/test/"
    recall7, precision7, f17 = test(traindir7, testdir7, r)
    print("序号7：")
    print("召回率(%)：", recall7)
    print("正确率(%)：", precision7)
    print("F1值(%)：", f17)
    traindir8 = "./data/email/ten_fold_cross/8/train/"
    testdir8 = "./data\email/ten_fold_cross/8/test/"
    recall8, precision8, f18 = test(traindir8, testdir8, r)
    print("序号8：")
    print("召回率(%)：", recall8)
    print("正确率(%)：", precision8)
    print("F1值(%)：", f18)
    traindir9 = "./data/email/ten_fold_cross/9/train/"
    testdir9 = "./data\email/ten_fold_cross/9/test/"
    recall9, precision9, f19 = test(traindir9, testdir9, r)
    print("序号9：")
    print("召回率(%)：", recall9)
    print("正确率(%)：", precision9)
    print("F1值(%)：", f19)
    traindir10 = "./data/email/ten_fold_cross/10/train/"
    testdir10 = "./data\email/ten_fold_cross/10/test/"
    recall10, precision10, f110 = test(traindir10, testdir10, r)
    print("序号10：")
    print("召回率(%)：", recall10)
    print("正确率(%)：", precision10)
    print("F1值(%)：", f110)
    recall_mean = (
                          recall1 + recall2 + recall3 + recall4 + recall5 + recall6 + recall7 + recall8 + recall9 + recall10) / 10
    precision_mean = (
                             precision1 + precision2 + precision3 + precision4 + precision5 + precision6 + precision7 + precision8 + precision9 + precision10) / 10
    f1_mean = (
                      f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19 + f110) / 10
    print("最终结果：")
    print("平均召回率(%)：", recall_mean)
    print("平均正确率(%)：", precision_mean)
    print("平均F1值(%)：", f1_mean)
