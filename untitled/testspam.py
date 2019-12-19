import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import warnings

warnings.filterwarnings("ignore")  # 过滤warnings

sms = pd.read_csv('./input/spam.csv', encoding='latin-1')  # 返回数据帧DataFrame格式的sms
sms = sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)  # 删除指定列，axis=1代表删的是列
sms = sms.rename(columns={'v1': 'label', 'v2': 'message'})  # 对列重新命名
text_feat = sms['message'].copy()  # 新的数据帧text_feat只存放各消息的内容


def text_process(text):  # 文本预处理，删除标点符号、进行分词、进行停用词过滤
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]  # 小写英文单词不在停用词表中
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")  # 基于Snowball 进行词干提取
        words += (stemmer.stem(i)) + " "
    return words


text_feat = text_feat.apply(text_process)  # 预处理之后的数据集文本
vectorizer = TfidfVectorizer("english")
train_data = vectorizer.fit_transform(text_feat)  # 得到各词的词频-逆文本频率，即TF-IDF
train_target = sms['label']  # 训练的目标就是邮件的标签
# 接下来分类和预测
# 首先将数据集分为训练集和测试集，避免过拟合，采用交叉验证，验证集占训练集30%，固定随机种子（random_state)
# train_data：所要划分的样本特征集;train_target：所要划分的样本结果
# 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
features_train, features_test, labels_train, labels_test = train_test_split(train_data, train_target, test_size=0.3,
                                                                            random_state=111)

# 1、Support VectorClassifier
print("******Support VectorClassifier******")
pred_scores = []
krnl = {'rbf': 'rbf', 'polynominal': 'poly', 'sigmoid': 'sigmoid'}
for k, v in krnl.items():
    for i in np.linspace(0.05, 1, num=20):  # num生成多少个数, 在全闭区间[0.05,1]
        svc = SVC(kernel=v, gamma=i)  # 得到模型
        svc.fit(features_train, labels_train)  # 训练模型
        pred = svc.predict(features_test)  # 预测测试样本的结果
        pred_scores.append((k, [i, accuracy_score(labels_test, pred)]))  # 用真实标签和预测结果计算分类准确率，并将数据保存
df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Gamma', 'Score'])  # 形成数据帧
df['Score'].plot(kind='line', figsize=(11, 6), ylim=(0.8, 1.0))  # 绘画折线图，y轴坐标范围[0.8,1.0]，代表Score
plt.ylabel('Accuracy Score')  # y轴坐标为精确率
plt.title('Support VectorClassifier')  # 图标题为Support VectorClassifier
# plt.show()
plt.savefig('./output/SVC.png')  # 保存折线图
a = df[(df.Score == df.loc[:, 'Score'].max())].index.tolist()  # 准确率最大的kernel的值
b = df[(df.Score == df.loc[:, 'Score'].max())].Gamma.tolist()  # 准确率最大的gamma的值
print("kernel:", a[0])
print("gamma:", b[0])
print("Accuracy Score:", df.loc[:, 'Score'].max())  # 输出最大准确率

# 2、K-Neighbours Classifier
print("******K-Neighbours Classifier******")
pred_scores = []
for i in range(3, 60):  # 迭代近邻的个数
    knc = KNeighborsClassifier(n_neighbors=i)  # 得到模型
    knc.fit(features_train, labels_train)  # 训练模型
    pred = knc.predict(features_test)  # 预测测试样本的结果
    pred_scores.append((i, [accuracy_score(labels_test, pred)]))  # 用真实标签和预测结果计算分类准确率，并将数据保存
df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Score'])  # 形成数据帧
df.plot(figsize=(11, 6))  # 绘画折线图
plt.ylabel('Accuracy Score')  # y轴坐标为精确率
plt.title('K-Neighbours Classifier')  # 图标题为K-Neighbours Classifier
# plt.show()
plt.savefig('./output/KN.png')  # 保存折线图
a = df[(df.Score == df.loc[:, 'Score'].max())].index.tolist()  # 准确率最大的K的值
print("K:", a[0])
print("Accuracy Score:", df.loc[:, 'Score'].max())  # 输出最大准确率

# 3、NaiveBayes Classifier
print("******NaiveBayes Classifier******")
pred_scores = []
for i in np.linspace(0.05, 1, num=20):  # alpha为一个大于0的常数，取1时即为拉普拉斯平滑；也可以取其他值。此处进行迭代
    mnb = MultinomialNB(alpha=i)  # 得到模型
    mnb.fit(features_train, labels_train)  # 训练模型
    pred = mnb.predict(features_test)  # 预测测试样本的结果
    pred_scores.append((i, [accuracy_score(labels_test, pred)]))  # 用真实标签和预测结果计算分类准确率，并将数据保存
df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Score'])  # 形成数据帧
df.plot(figsize=(11, 6))  # 绘画折线图
plt.ylabel('Accuracy Score')  # y轴坐标为精确率
plt.title('NaiveBayes Classifier')  # 图标题为NaiveBayes Classifier
# plt.show()
plt.savefig('./output/NB.png')  # 保存折线图
a = df[(df.Score == df.loc[:, 'Score'].max())].index.tolist()  # 准确率最大的alpha的值
print("NB:", a[0])
print("Accuracy Score:", df.loc[:, 'Score'].max())  # 输出最大准确率

# 4、Decision Tree Classifier
print("******Decision Tree Classifier******")
pred_scores = []
for i in range(2, 21):
    dtc = DecisionTreeClassifier(min_samples_split=i, random_state=111)  # 得到模型
    dtc.fit(features_train, labels_train)  # 训练模型
    pred = dtc.predict(features_test)  # 预测测试样本的结果
    pred_scores.append((i, [accuracy_score(labels_test, pred)]))  # 用真实标签和预测结果计算分类准确率，并将数据保存
df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Score'])  # 形成数据帧
df.plot(figsize=(11, 6))  # 绘画折线图
plt.ylabel('Accuracy Score')  # y轴坐标为精确率
plt.title('Decision Tree Classifier')  # 图标题为Decision Tree Classifier
# plt.show()
plt.savefig('./output/DTC.png')  # 保存折线图
a = df[(df.Score == df.loc[:, 'Score'].max())].index.tolist()
print("DT:", a[0])
print("Accuracy Score:", df.loc[:, 'Score'].max())  # 输出最大准确率

# 5、LogisticRegression
print("******LogisticRegression Classifier******")
slvr = {'newton-cg': 'newton-cg', 'lbfgs': 'lbfgs', 'liblinear': 'liblinear', 'sag': 'sag'}
pred_scores = []
for k, v in slvr.items():
    lrc = LogisticRegression(solver=v, penalty='l2')  # 得到模型
    lrc.fit(features_train, labels_train)  # 训练模型
    pred = lrc.predict(features_test)  # 预测测试样本的结果
    pred_scores.append((k, [accuracy_score(labels_test, pred)]))  # 用真实标签和预测结果计算分类准确率，并将数据保存
df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Score'])  # 形成数据帧
df.plot(figsize=(11, 6))  # 绘画折线图
plt.ylabel('Accuracy Score')  # y轴坐标为精确率
plt.title('LogisticRegression Classifier')  # 图标题为LogisticRegression Classifier
# plt.show()
plt.savefig('./output/LRC.png')  # 保存折线图
a = df[(df.Score == df.loc[:, 'Score'].max())].index.tolist()
print("LR:", a[0])
print("Accuracy Score:", df.loc[:, 'Score'].max())  # 输出最大准确率
pred_scores = []
lrc = LogisticRegression(solver='liblinear', penalty='l1')
lrc.fit(features_train, labels_train)
pred = lrc.predict(features_test)
print(accuracy_score(labels_test, pred))

# 6、Adaboost Classifier
print("******Adaboost Classifier******")
pred_scores = []
for i in range(25, 76):
    abc = AdaBoostClassifier(n_estimators=i, random_state=111)  # 得到模型
    abc.fit(features_train, labels_train)  # 训练模型
    pred = abc.predict(features_test)  # 预测测试样本的结果
    pred_scores.append((i, [accuracy_score(labels_test, pred)]))  # 用真实标签和预测结果计算分类准确率，并将数据保存
df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Score'])  # 形成数据帧
df.plot(figsize=(11, 6))
plt.ylabel('Accuracy Score')  # y轴坐标为精确率
plt.title('Adaboost Classifier')  # 图标题为Adaboost Classifier
# plt.show()
plt.savefig('./output/ADC.png')  # 保存折线图
a = df[(df.Score == df.loc[:, 'Score'].max())].index.tolist()
print("Adaboost:", a[0])
print("Accuracy Score:", df.loc[:, 'Score'].max())  # 输出最大准确率
