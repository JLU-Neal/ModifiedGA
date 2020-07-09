# 线性回归

import numpy as np  # 快速操作结构数组的工具

import matplotlib.pyplot as plt  # 可视化绘制

from sklearn.linear_model import LinearRegression  # 线性回归

from mpl_toolkits.mplot3d import Axes3D  # 绘制3D点坐标
from mlrGAtsne.TSNE import TSNE
from mlrGAtsne.Filter import Filter
from mlrGAtsne.Configuration import Configuration


class MLR():

    def __init__(self):
        self.data = np.array([])
        self.model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
        self.embeddedX=np.array([])
        self.filter=Filter()

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



    def train(self,X,datasize,con:Configuration):
        # dataMat = np.array(data)
        temp = self.data[self.data.shape[0] - datasize:,0:self.data.shape[1]-1]
        temp = np.vstack((temp, X))

        n_components = 3
        if con.mode!='mlr_sga':
            #当需要进行降维时
            tsne = TSNE(n_components,con)
            self.embeddedX = tsne.transform(temp)
        else:
            #当不需要降维时
            self.embeddedX=temp



        X = self.embeddedX[0:self.embeddedX.shape[0]-X.shape[0], :]  # 变量x

        y = self.data[self.data.shape[0]-datasize:, self.data.shape[1]-1 ]  # 变量y

        # ========线性回归========
        #加入随机筛选
        [X, y] = self.filter.randomFilter(X, y)
        self.model.fit(X, y)  # 线性回归建模

        #print('系数矩阵:\n', self.model.coef_)

        #print('线性回归模型:\n', self.model)


    def train_with_filter(self,X,datasize,con:Configuration):
        # dataMat = np.array(data)
        temp = self.data[self.data.shape[0] - datasize:, :]
        #temp=self.filter.APFilter(temp)


        n_components = 3
        # 当需要进行降维时
        """
        
        """
        tsne = TSNE(n_components,con)
        self.embeddedX = tsne.transform(np.vstack((temp[:, :temp.shape[1] - 1], X)))
        X = self.embeddedX[0:self.embeddedX.shape[0] - X.shape[0], :]
        y = temp[:, temp.shape[1] - 1]  # 变量y
        [X,y]=self.filter.APFilter(X,y)
        #当不需要进行降维时
        """
        self.embeddedX=np.vstack((temp[:,:temp.shape[1]-1],X))
        X = temp[:,:temp.shape[1]-1]  # 变量x

        y = temp[:,temp.shape[1]-1]  # 变量y
        """


        # ========线性回归========

        self.model.fit(X, y)  # 线性回归建模

        # print('系数矩阵:\n', self.model.coef_)

        # print('线性回归模型:\n', self.model)

    def predict(self, predictsize):
        predict_set=self.embeddedX[self.embeddedX.shape[0]-predictsize:,:]
        predicted = self.model.predict(predict_set)
        return predicted


