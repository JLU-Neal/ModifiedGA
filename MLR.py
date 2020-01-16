# 线性回归

import numpy as np  # 快速操作结构数组的工具

import matplotlib.pyplot as plt  # 可视化绘制

from sklearn.linear_model import LinearRegression  # 线性回归

from mpl_toolkits.mplot3d import Axes3D#绘制3D点坐标


"""
# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型

data = [

    [0.067732, 3.176513,1], [0.427810, 3.816464,1], [0.995731, 4.550095,1], [0.738336, 4.256571,1], [0.981083, 4.560815,1],

    [0.526171, 3.929515,2], [0.378887, 3.526170,2], [0.033859, 3.156393,2], [0.132791, 3.110301,2], [0.138306, 3.149813,2],

    [0.247809, 3.476346,3], [0.648270, 4.119688,3], [0.731209, 4.282233,3], [0.236833, 3.486582,3], [0.969788, 4.655492,3],

    [0.607492, 3.965162,4], [0.358622, 3.514900,4], [0.147846, 3.125947,4], [0.637820, 4.094115,4], [0.230372, 3.476039,4],

    [0.070237, 3.210610,5], [0.067154, 3.190612,5], [0.925577, 4.631504,5], [0.717733, 4.295890,5], [0.015371, 3.085028,5],

    [0.335070, 3.448080,6], [0.040486, 3.167440,6], [0.212575, 3.364266,6], [0.617218, 3.993482,6], [0.541196, 3.891471,6]

]
"""

class MLR():

    def __int__(self,data):
        # 生成X和y矩阵

        dataMat = np.array(data)

        X = dataMat[:, 0:2]  # 变量x

        y = dataMat[:, 2]  # 变量y

        # ========线性回归========

        self.model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)

        self.model.fit(X, y)  # 线性回归建模

        print('系数矩阵:\n', self.model.coef_)

        print('线性回归模型:\n', self.model)

        # 使用模型预测

        predicted = self.model.predict(X)

        # 绘制散点图 参数：x横轴 y纵轴
        """
        plt.scatter(X, y, marker='x')

        plt.plot(X, predicted, c='r')

        # 绘制x轴和y轴坐标

        plt.xlabel("x")

        plt.ylabel("y")

        # 显示图形
        """
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='^')  # 点为红色三角形
        ax.scatter(X[:, 0], X[:, 1], predicted)  # 点为红色三角形
        ax.plot_trisurf(X[:, 0], X[:, 1], predicted)
        plt.show()


    def predict(self,predict_set):
        predicted=self.model.predict(predict_set)
        return predicted
