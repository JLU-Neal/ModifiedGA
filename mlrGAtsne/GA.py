# -*- coding: utf-8 -*-
""" QuickStart """
import numpy as np
import geatpy as ea
from mlrGAtsne.MLRwithTSNE import MLR
from mlrGAtsne.FitnessFun import FitnessFun
from mlrGAtsne.Configuration import Configuration
from matplotlib import pyplot as plt
# 自定义问题类
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self,con:Configuration):
        self.mlr=MLR()
        self.current_generation=0 #用于记录当前进化代数
        self.fitfun=FitnessFun()
        #self.split=10
        self.split=con.split
        #self.datasize=100
        self.datasize=con.datasize
        self.con=con
        self.fitnessvalue=[]
        self.regressionvalue=[]
        print("class been created1")
        name = 'DTLZ1'  # 初始化name（函数名称，可以随意设置）
        M=1
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        #Dim = 30  # 初始化Dim（决策变量维数）
        Dim=con.Dim
        varTypes = np.array([0] * Dim)  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        #lb = [-100] * Dim  # 决策变量下界
        lb = [con.lb] * Dim
        #ub = [100] * Dim  # 决策变量上界
        ub = [con.ub] * Dim  # 决策变量上界

        #lbin = [1] * Dim  # 决策变量下边界
        lbin = [con.lbin] * Dim
        #ubin = [1] * Dim  # 决策变量上边界
        ubin = [con.ubin] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        self.current_generation+=1
        print(self.current_generation)
        swich_mode_generation=10000


        if self.con.mode=='sga':

           # 未使用代理模式
            X = pop.Phen
            #Y = self.fitfun.Step(X)
            Y = self.fitfun.chooseFunction(X,self.con.function)
            pop.ObjV = Y
            """

            """
            # 用于保存每一代最佳个体
            # comparision = self.fitfun.Step(X)
            comparision = self.fitfun.chooseFunction(X, self.con.function)

            min = 999999999999999999
            for element in comparision:
                if element < min:
                    min = element
            self.fitnessvalue.append(min)
            if min > self.fitnessvalue[len(self.fitnessvalue) - 2]:
                self.fitnessvalue[len(self.fitnessvalue) - 1] = self.fitnessvalue[len(self.fitnessvalue) - 2]

            if self.fitnessvalue[len(self.fitnessvalue) - 1] > self.fitnessvalue[len(self.fitnessvalue) - 2]:
                print("fuck")



        else:
            if self.current_generation > 0 and self.current_generation < swich_mode_generation:

                # 测试sphere函数
                X = pop.Phen  # 得到决策变量矩阵


                if self.mlr.data.shape[0] < self.datasize or self.mlr.data.size == 0:  # 当数据量不足以进行训练的话，先用适应度函数进行计算
                    Y = self.fitfun.chooseFunction(X,self.con.function)
                    pop.ObjV = Y
                    self.mlr.insert(X, Y)
                    # print("PHASE I")
                elif self.mlr.data.shape[0] >= self.datasize:
                    #Y = self.fitfun.Step(X[0:self.split, :])
                    Y = self.fitfun.chooseFunction(X[0:self.split, :],self.con.function)


                    sample = X[0:self.split, :]

                    self.mlr.insert(X[0:self.split, :], Y)
                    print("PHASE II")
                    if self.con.mode=='lle_ap_mlr_sga' or self.con.mode=='tsne_ap_mlr_sga'or'isomap_ap_mlr_sga':
                        # 使用过滤器
                         self.mlr.train_with_filter(X=X[self.split:,:],datasize=self.datasize,con=self.con)
                    else:
                        # 不使用过滤器
                        self.mlr.train(X=X[self.split:, :], datasize=self.datasize,con=self.con)

                    tempY = self.mlr.predict(pop.Phen.shape[0] - self.split)
                    tempY = np.array(tempY).reshape(pop.Phen.shape[0] - self.split, 1)
                    Y = np.vstack((Y, tempY))
                    pop.ObjV = Y
                    """
                    用于最后比较每一代的回归值与实际值
                    comparision=self.fitfun.cal(X)
                    min=999999999999999999
                    for element in comparision:
                        if element<min:
                            min=element
                    self.fitnessvalue.append(min)

                    min=999999999999999999
                    for element in Y:
                        if element<min:
                            min=element
                    self.regressionvalue.append(min)
                    if self.current_generation == 1499:
                        a=np.arange(497)
                        fig=plt.figure()
                        plt.plot(a,self.regressionvalue,label='Regression')
                        plt.plot(a,self.fitnessvalue,label='FitnessFun')
                        plt.legend()
                        plt.show(fig)
                    """

                    """
                    用于查看每一代的拟合效果
                    comparison = self.fitfun.cal(X)
                    print("??????????????????/")
                    print(Y)
                    print('================versus==============')
                    print(comparison)
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    a=np.arange(100)
                    fig=plt.figure()
                    plt.plot(a,Y,label='Regression')
                    plt.plot(a,comparison,label='FitnessFun')
                    plt.legend()
                    plt.show(fig)
                    print()
                    """

                """

                """
                # 用于保存每一代最佳个体
                # comparision = self.fitfun.Step(X)
                comparision = self.fitfun.chooseFunction(X, self.con.function)

                min = 999999999999999999
                for element in comparision:
                    if element < min:
                        min = element
                self.fitnessvalue.append(min)
                if min > self.fitnessvalue[len(self.fitnessvalue) - 2]:
                    self.fitnessvalue[len(self.fitnessvalue) - 1] = self.fitnessvalue[len(self.fitnessvalue) - 2]


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

