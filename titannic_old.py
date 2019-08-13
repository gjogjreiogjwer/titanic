# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

#(1)初探数据********************************************************
data_train = pd.read_csv('train.csv')
# 发现age和cabin数据缺失
#print (data_train.info())
#print (data_train.describe())

#(2)数据初步分析********************************************************
# 1.乘客各属性分布
import matplotlib.pyplot as plt
# 解决中文显示问题
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/Library/Fonts/Songti.ttc')

# 显示所有dataframe行列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

# fig = plt.figure()
# # 设定图标颜色alpha参数
# fig.set(alpha=0.2)

# # 在一张大图里分列几个小图
# plt.subplot2grid((2,3), (0,0))
# # 柱状图
# data_train.Survived.value_counts().plot(kind='bar')
# # 标题
# plt.title('获救情况')
# plt.ylabel('人数')

# plt.subplot2grid((2,3), (0,1))
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.title('乘客分布等级')
# plt.ylabel('人数')

# plt.subplot2grid((2,3), (0,2))
# # 散点图
# plt.scatter(data_train.Survived, data_train.Age)
# # 网格线设置   b：是否显示网格线，axis：绘制哪个方向的网格线(x,y,both)
# plt.grid(b=True, axis='y')
# plt.ylabel('年龄')
# plt.title('按年龄看获救分布')

# # 行跨度为2，列跨度是rowspan
# plt.subplot2grid((2,3), (1,0), colspan=2)
# # 密度图,label用于图例
# data_train.Age[data_train.Pclass == 1].plot(kind='kde', label='头等舱')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde', label='2等舱')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde', label='3等舱')
# plt.xlabel('年龄')
# plt.ylabel('密度')
# plt.title('各等级的乘客年龄分布')
# # legend()用于显示图例,'best':位置自适应
# plt.legend(loc='best')

# plt.subplot2grid((2,3), (1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title('各登船口上船人数')
# plt.ylabel('人数')

# #2.属性与获救结果的关联联系
# #各等级的获救情况
# survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
# survived_1 = data_train.Pclass[data_train.Survived==1].value_counts()
# # DataFrame是一种二维表
# df = pd.DataFrame({'获救':survived_1, '未获救':survived_0})
# #print (df)
# df.plot(kind='bar', stacked=True)
# plt.title('各等级获救情况')
# plt.xlabel('乘客等级')
# plt.ylabel('人数')

# #各性别的获救情况
# survived_0 = data_train.Sex[data_train.Survived==0].value_counts()
# survived_1 = data_train.Sex[data_train.Survived==1].value_counts()
# # DataFrame是一种二维表
# df = pd.DataFrame({'获救':survived_1, '未获救':survived_0})
# print (df)
# df.plot(kind='bar', stacked=True)
# plt.title('各性别获救情况')
# plt.xlabel('性别')
# plt.ylabel('人数')

# # 各舱级各性别获救情况
# fig = plt.figure()
# plt.title('根据舱等级和性别的获救情况')

# ax1 = fig.add_subplot(141)
# data_train.Survived[data_train.Sex=='female'][data_train.Pclass!=3].value_counts().plot(kind='bar', label='女性/高级舱')
# ax1.set_xticklabels(['获救', '未获救'], rotation=0)
# ax1.legend(['女性/高级舱'], loc='best')

# #142:一行四列，第二个图，与ax1共用y坐标
# ax2 = fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar', label='女性/低级舱')
# ax2.set_xticklabels(['获救', '未获救'], rotation=0)
# ax2.legend(['女性/低级舱'], loc='best')

# ax3 = fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex=='male'][data_train.Pclass!=3].value_counts().plot(kind='bar', label='男性/高级舱')
# ax3.set_xticklabels(['获救', '未获救'], rotation=0)
# ax3.legend(['男性/高级舱'], loc='best')

# ax4 = fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar', label='男性/低级舱')
# ax4.set_xticklabels(['获救', '未获救'], rotation=0)
# ax4.legend(['男性/低级舱'], loc='best')

# #各登船港口的获救情况
# survived_0 = data_train.Embarked[data_train.Survived==0].value_counts()
# survived_1 = data_train.Embarked[data_train.Survived==1].value_counts()
# # DataFrame是一种二维表
# df = pd.DataFrame({'获救':survived_1, '未获救':survived_0})
# print (survived_0)
# print (pd.DataFrame(survived_0))
# print (df)
# df.plot(kind='bar', stacked=True)
# plt.title('各登船港口获救情况')
# plt.xlabel('登船港口')
# plt.ylabel('人数')

#plt.show()

# # groupby：按照（）分组
# g = data_train.groupby(['SibSp', 'Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print (df)

# g = data_train.groupby(['Parch', 'Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print (df)

# # cabin只有214个乘客有值,观察数据不集中
# print (data_train.Cabin.value_counts())
# # 先把cabin有无当成条件
# fig = plt.figure()
# survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df = pd.DataFrame({'有':survived_cabin, '无':survived_nocabin})
# df.plot(kind='bar', stacked=True)
# plt.title('按cabin有无看获救情况')
# plt.xlabel('获救情况')
# plt.ylabel('人数')
# plt.show()

'''
(3)数据预处理********************************************************
对缺失样本的处理：
	1.如果缺失样本占总数比例极高，可能直接舍弃
	2.如果缺失值适中，而该特征为非连续属性，那就吧NaN（not a number）作为一个新类别
	3.如果缺失值适中，而该特征为连续属性，给定一个step（e.g. age,每隔2/3岁为一步长），
	  然后把它离散化，之后把NaN作为一个type加到属性中
	4.如果缺失值不是很多，根据已有的值，拟合数据
age数据缺失，使用RandomForest来拟合缺失的年龄数据
cabin数据缺失，把数据有无作为特征
'''
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
	# 把已有的数值型特征取出来放入随机森林回归中
	age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
	# 乘客分为已知年龄和未知年龄两部分
	known_age = age_df[age_df.Age.notnull()].as_matrix()
	unknown_age = age_df[age_df.Age.isnull()].as_matrix()
	# 目标年龄
	y = known_age[:, 0]
	# 其他特征
	x = known_age[:, 1:]
	rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
	rfr.fit(x, y)
	# 用得到的模型进行未知年龄结果预测
	predictedAges = rfr.predict(unknown_age[:, 1:])
	# 用得到的预测结果填补原缺失数据
	df.loc[(df.Age.isnull()), 'Age'] = predictedAges
	return df, rfr

def set_Cabin_type(df):
	df.loc[(df.Cabin.notnull()), 'Cabin'] = 'yes'
	df.loc[(df.Cabin.isnull()), 'Cabin'] = 'no'
	return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

# 特征因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
# print (data_train['Cabin'])
# print (dummies_Cabin)
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
# 合并
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
# print (dummies_Sex)
# print (data_train)
# print (df)
# print (df['Sex_male'])
# 删除原先不是数值型的特征
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#print (df)

# Age,Fare数值变化幅度太大，对数据进行缩放scaling，[-1,1]之间
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)
#print (df)

#(4)逻辑回归建模********************************************************
from sklearn import linear_model

#用正则表达式取出想要的特征
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:, 0]
x = train_np[:, 1:]

# fit到RandomForestRegressor中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(x, y)

#print (clf)

# 对test_data做预处理
data_test = pd.read_csv('test.csv')
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

data_test, rfr = set_missing_ages(data_test)
data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)
#print (df_test)

# 预测结果
test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
#print (predictions)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
#print (result)
#result.to_csv('predictions.csv', index=False)


# 查看错误率
def errorRate(filename):
	classifierResult = pd.read_csv(filename).as_matrix()
	label = pd.read_csv('gender_submission.csv').as_matrix()
	test = pd.read_csv('test.csv').as_matrix().tolist()
	errorCount = 0
	m = len(classifierResult)
	n = len(test[0])
	badCases = []
	for i in range(m):
		if classifierResult[i, 1] != label[i, 1]:
			errorCount += 1
			# current = []
			# for j in range(n):
			# 	current.append(test[i][j])
			# badCases.append(current)
			badCases.append(test[i])
	print ("error rate: ", errorCount/m)
	return badCases

#print (errorRate('predictions.csv'))


#(5)逻辑回归优化********************************************************

# 1.模型系数关联分析，系数为正的特征和结果正相关，反之反相关
# print (pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))

# 2.交叉验证
# 把train.csv分为两部分，一部分用于训练，另一部分用于预测算法的效果
from sklearn import model_selection
#print (model_selection.cross_val_score(clf, x, y, cv=5))

# 分割数据，训练数据：测试数据 7:3
split_train, split_cv = model_selection.train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

# 对cross validation数据进行预测
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.as_matrix()[:, 1:])

# 查看分错的cases
origin_data_train = pd.read_csv('train.csv')
bad_cases = origin_data_train.loc[origin_data_train.PassengerId.isin(split_cv[predictions != cv_df.as_matrix()[:,0]].PassengerId.values)]
# print (bad_cases)
print ('*************************************', len(bad_cases))


# 3.learning curves   用于判断处于欠拟合或者过拟合
# 样本数为横坐标，训练和交叉验证集上的错误率(或准确率)作为纵坐标

'''
用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
Args:
	estimator: 分类器
	title: 表格标题
	x: 输入的feature
	y: 输入的target
	ylim: 设定图像纵坐标的最低点和最高点
	cv: 做交叉验证时，数据分成的份数，默认为3份
	n_jobs: 并行的任务数
'''
def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=1,train_size=np.linspace(0.05,1,20), verbose=0, plot=True):
	train_size, train_scores, test_scores = model_selection.learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs,train_sizes=train_size, verbose=verbose)
	
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	if plot:
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel('训练样本数')
		plt.ylabel('准确率')
		plt.grid()

		plt.fill_between(train_size, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='b')
		plt.fill_between(train_size, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='r')
		plt.plot(train_size, train_scores_mean, 'o-', color='b', label='训练集上得分')
		plt.plot(train_size, test_scores_mean, 'o-', color='r', label='交叉验证集上得分')

		plt.legend(loc='best')

		plt.draw()
		plt.show()

	midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
	diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
	return midpoint, diff
# plot_learning_curve(clf, '学习曲线', x, y)
'''
通过图片看出处于欠拟合，再做一些feature engineering
	1.Age属性不使用现在的拟合方式，而是根据名称中的『Mr』『Mrs』『Miss』等的平均值进行填充。
	2.Age不做成一个连续值属性，而是使用一个步长进行离散化，变成离散的类目feature。
	3.Cabin再细化一些，对于有记录的Cabin属性，我们将其分为前面的字母部分(我猜是位置和船层之类的信息) 和 后面的数字部分(应该是房间号，有意思的事情是，如果你仔细看看原始数据，你会发现，这个值大的情况下，似乎获救的可能性高一些)。
	4.Pclass和Sex俩太重要了，我们试着用它们去组出一个组合属性来试试，这也是另外一种程度的细化。
	5.单加一个Child字段，Age<=12的，设为1，其余为0(你去看看数据，确实小盆友优先程度很高啊)
	6.如果名字里面有『Mrs』，而Parch>1的，我们猜测她可能是一个母亲，应该获救的概率也会提高，因此可以多加一个Mother字段，此种情况下设为1，其余情况下设为0
	7.登船港口可以考虑先去掉试试(Q和C本来就没权重，S有点诡异)
	8.把堂兄弟/兄妹 和 Parch 还有自己 个数加在一起组一个Family_size字段(考虑到大家族可能对最后的结果有影响)
	9.Name是一个我们一直没有触碰的属性，我们可以做一些简单的处理，比如说男性中带某些字眼的(‘Capt’, ‘Don’, ‘Major’, ‘Sir’)可以统一到一个Title，女性也一样。
	最好的结果是在『Survived~C(Pclass)+C(Title)+C(Sex)+C(Age_bucket)+C(Cabin_num_bucket)Mother+Fare+Family_Size』下取得的
'''


'''
(6)模型融合********************************************************
手头上有一堆在同一份数据集上训练得到的分类器(比如logistic regression，SVM，KNN，random forest，
神经网络)，那我们让他们都分别去做判定，然后对结果做投票统计，取票数最多的结果为最后结果。
模型融合可以很好地解决过拟合问题。
如何只通过逻辑回归，应用融合思想
每次训练集取一个subset，做训练，即使出现过拟合，也是在子训练集上过拟合
'''
from sklearn .ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:, 0]
x = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(x, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
#result.to_csv('predictions1.csv', index=False)

print (df.columns)

