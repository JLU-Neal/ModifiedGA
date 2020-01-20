# -*- coding: utf-8 -*-
""" QuickStart """
import numpy as np
import geatpy as ea
from MLR import MLR

# 自定义问题类
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'DTLZ1'  # 初始化name（函数名称，可以随意设置）
        M=1
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 10  # 初始化Dim（决策变量维数）
        varTypes = np.array([0] * Dim)  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1000] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        """
        Vars = pop.Phen # 得到决策变量矩阵
        XM = Vars[:,(self.M-1):]
        g = 100 * (self.Dim - self.M + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1, keepdims = True))
        ones_metrix = np.ones((Vars.shape[0], 1))
        f = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_metrix, Vars[:,:self.M-1]]), 1)) * np.hstack([ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]]) * np.tile(1 + g, (1, self.M))
        pop.ObjV = f # 把求得的目标函数值赋值给种群pop的ObjV
        #插入机器学习算法
        print(pop.ObjV)
        print("hello")
        """
        # 测试sphere函数
        Vars = pop.Phen  # 得到决策变量矩阵
        """
        x0 = Vars[:, [0]]
        x1 = Vars[:, [1]]
        x2 = Vars[:, [2]]
        x3 = Vars[:, [3]]
        x4 = Vars[:, [4]]
        x5 = Vars[:, [5]]
        x6 = Vars[:, [6]]
        x7 = Vars[:, [7]]
        x8 = Vars[:, [8]]
        x9 = Vars[:, [9]]
        """


        #matrix = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 * +x5 * x5 + x6 * x6 + x7 * x7 + x8 * x8 + x9 * x9
        for index in range(Vars.shape[1]):
            #print(index)
            #x=Vars[:,[index]]*Vars[:,[index]]
            Vars[:, [index]] *= Vars[:, [index]]#此处错误，应该把每列值加起来。
        #print(Vars)
        Y=0#Result
        for index in range(Vars.shape[1]):
            Y=Y+Vars[:,[index]]
        #mlr=MLR(Y)#改如何添加染色体？

        pop.ObjV = Y
        # print(pop.Phen)
        #print(pop.ObjV)
        """
        for index in range(Vars.shape[1]):
            # print(index)

            Vars[:, [index]] *= Vars[:, [index]]

            # print(Vars)

        pop.ObjV = Vars
        """



    def calBest(self):  # 计算全局最优解
        uniformPoint, ans = ea.crtup(self.M, 10000)  # 生成10000个在各目标的单位维度上均匀分布的参考点
        globalBestObjV = uniformPoint / 2
        return globalBestObjV

