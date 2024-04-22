from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from unzip import unzip
import matplotlib.pyplot as plt
import os

# unzip the file to a folder called "Data"
try:
    if os.path.exists("Data"):
        print("The file is already unzipped")
except:
    unzip("titanic.zip", "Data")
import pandas as pd

# read the csv file
train_data = pd.read_csv('./Data/train.csv')
# train_data.hist()
# show the histogram of the data
# plt.show()

# from the datainfo we could know that there are some missing values in the data
## Age, Cabin, Embarked have missing values
### we need to fill the missing values
train_data.describe()
# PassengerId,cabine,embarked is not useful for the prediction
train_data.drop(['PassengerId', 'Cabin', 'Embarked'], axis=1, inplace=True)
# fill the missing values
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data.info()
# target is the Survived column
Y = train_data['Survived']
X = train_data.drop('Survived', axis=1)


class NBClassifier(object):
    def __init__(self):
        self.x = []
        self.y = []
        self.py = defaultdict(int)  # P(y)
        self.pxy = defaultdict(lambda: defaultdict(list))  # P(x|y)
        self.px = defaultdict(int)
        self.n = 5

    def step(self, x, n):
        return (x - x.min()) / (x.max() - x.min())
        # def step(self, arr, n):
        #     ma = max(arr)
        #     mi = min(arr)
        #     step_size = (ma - mi) / self.n
        #     return np.array([(x - mi) // step_size + 1 for x in arr])
        # def step(self, arr, n):
        """
        # 分为n阶
        # """
        # ma = max(arr)
        # mi = min(arr)
        # for i in range(len(arr)):
        #     for j in range(n):
        #         a = mi + (ma - mi) * (j / n)
        #         b = mi + (ma - mi) * ((j + 1) / n)
        #         if a <= arr[i] <= b:
        #             arr[i] = j + 1
        #             break

    def processor(self, x):
        encoder = LabelEncoder()
        # x = x.apply(encoder.fit_transform)
        for col in x.columns:
            if x[col].dtype == "object":  # 和直接col.dtype 有啥区别？
                x[col] = encoder.fit_transform(x[col])
                x[col] = self.step(x[col].values, self.n)
            elif x[col].dtype == "float64" or x[col].dtype == "int64":
                x[col] = self.step(x[col].values, self.n)
        return x

    def get_set(self, x, y):
        self.y = list(set(y))
        for i in range(x.shape[1]):
            self.x.append(list(set(x[:, i])))  # x[:, i] 是第i列的所有值 set(x[:, i])是第i列的所有值的不重复的值,为什么要不重复的值呢？
            # 因为我们要计算P(x|y)的概率，如果有重复的值，那么我们就要计算重复的值的概率，这样就会增加计算的复杂度

    def fit(self, x, y):
        x = self.processor(x)
        self.get_set(x.values, y.values)
        # p(y)
        for i in self.y:
            self.py[i] = (y == i).sum() / len(y)
        # p(x|y)
        for i in range(x.shape[1]):
            for j in self.y:
                for k in self.x[i]:
                    self.pxy[j][i].append(((x[y == j].iloc[:, i] == k).sum() + 1) / (y == j).sum() + len(self.x[i]))
        # p(x)
        for i in range(x.shape[1]):
            for j in self.x[i]:
                self.px[i] = (x.iloc[:, i] == j).sum() / len(x)




# test
class1 = NBClassifier()
# process X
X = class1.processor(X)
print(X)
