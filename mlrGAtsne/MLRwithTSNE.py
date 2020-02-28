# 线性回归

import numpy as np  # 快速操作结构数组的工具

import matplotlib.pyplot as plt  # 可视化绘制

from sklearn.linear_model import LinearRegression  # 线性回归

from mpl_toolkits.mplot3d import Axes3D  # 绘制3D点坐标
from mlrGAtsne.TSNE import TSNE



class MLR():

    def __init__(self):
        self.data = np.array([])
        self.model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
        self.embeddedX=0

    def insert(self, X, Y,split,datasize):  # 插入染色体
        new_chrome = np.hstack((X[0:split,:], Y))#将已经通过适应度函数计算的分离出来
        # data = np.vstack((self.data, new_chrome))

        if self.data.size == 0:
            self.data = new_chrome
        else:
            self.data = np.vstack((self.data, new_chrome))
            if self.data.shape[0]>=datasize:
                temp=self.data[self.data.shape[0]-datasize-1:self.data.shape[0]]
                temp=np.vstack((temp,X[split:X.size[0],:]))
                n_components=20
                tsne=TSNE(n_components)
                self.embeddedX=tsne.transform(temp)

    def insert(self,X,Y):
        new_chrome = np.hstack((X, Y))  # 将已经通过适应度函数计算的分离出来
        # data = np.vstack((self.data, new_chrome))

        if self.data.size == 0:
            self.data = new_chrome
        else:
            self.data = np.vstack((self.data, new_chrome))



    def train(self,X,datasize):
        # dataMat = np.array(data)
        temp = self.data[self.data.shape[0] - datasize:,0:self.data.shape[1]-1]
        temp = np.vstack((temp, X))
        n_components = 3
        #当需要进行降维时
        #tsne = TSNE(n_components)
        #self.embeddedX = tsne.transform(temp)
        #当不需要降维时
        self.embeddedX=temp



        X = self.embeddedX[0:self.embeddedX.shape[0]-X.shape[0], :]  # 变量x

        y = self.data[self.data.shape[0]-datasize:, self.data.shape[1]-1 ]  # 变量y

        # ========线性回归========

        self.model.fit(X, y)  # 线性回归建模

        #print('系数矩阵:\n', self.model.coef_)

        #print('线性回归模型:\n', self.model)


    def predict(self, datasize):
        predict_set=self.embeddedX[datasize:,:]
        predicted = self.model.predict(predict_set)
        return predicted


