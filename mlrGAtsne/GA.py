# -*- coding: utf-8 -*-
""" QuickStart """
import numpy as np
import geatpy as ea
from mlrGAtsne.MLRwithTSNE import MLR
from mlrGAtsne.FitnessFun import FitnessFun

# 自定义问题类
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        self.mlr=MLR()
        self.current_generation=0 #用于记录当前进化代数
        self.fitfun=FitnessFun()
        self.split=25
        self.datasize=500
        print("class been created1")
        name = 'DTLZ1'  # 初始化name（函数名称，可以随意设置）
        M=1
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 30  # 初始化Dim（决策变量维数）
        varTypes = np.array([0] * Dim)  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1000] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        self.current_generation+=1
        print(self.current_generation)
        swich_mode_generation=500

        if self.current_generation>0 and self.current_generation<swich_mode_generation:

            # 测试sphere函数
            X = pop.Phen  # 得到决策变量矩阵
            """
            # 根据函数要求，每个向量中的各个元素进行平方并累加可得函数值
            X = Vars
            for index in range(Vars.shape[1]):
                # print(index)
                # x=Vars[:,[index]]*Vars[:,[index]]
                Vars[:, [index]] *= Vars[:, [index]]  # 进行平方
            # print(Vars)
            Y = 0  # Result
            for index in range(Vars.shape[1]):
                Y = Y + Vars[:, [index]]  # 进行累加
            # mlr=MLR(Y)#改如何添加染色体？
            """

            if self.mlr.data.shape[0]<self.datasize or self.mlr.data.size==0:#当数据量不足以进行训练的话，先用适应度函数进行计算
                Y=self.fitfun.cal(X)
                pop.ObjV = Y
                self.mlr.insert(X, Y)
                print("PHASE I")
            elif self.mlr.data.shape[0]>=self.datasize:
                Y = self.fitfun.cal(X[0:self.split, :])
                self.mlr.insert(X[0:self.split,:],Y)
                print("PHASE II")
                self.mlr.train(X[self.split:,:],self.datasize)
                tempY=self.mlr.predict(self.datasize)
                tempY=np.array(tempY).reshape(75,1)
                Y=np.vstack((Y,tempY))
                pop.ObjV=Y





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

