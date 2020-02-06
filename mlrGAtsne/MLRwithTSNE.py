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


    def insert(self, X, Y):  # 插入染色体
        new_chrome = np.hstack((X, Y))
        # data = np.vstack((self.data, new_chrome))
        if self.data.size == 0:
            self.data = new_chrome
        else:
            self.data = np.vstack((self.data, new_chrome))

    def train(self):
        # dataMat = np.array(data)
        dataMat = self.data

        n_components = 3
        tsne=TSNE(n_components)
        dataMat=tsne.transform(dataMat)


        X = dataMat[:, 0:dataMat.shape[1] - 1]  # 变量x

        y = dataMat[:, dataMat.shape[1] - 1]  # 变量y

        # ========线性回归========

        self.model.fit(X, y)  # 线性回归建模

        print('系数矩阵:\n', self.model.coef_)

        print('线性回归模型:\n', self.model)


    def predict(self, predict_set):
        predicted = self.model.predict(predict_set)
        return predicted
