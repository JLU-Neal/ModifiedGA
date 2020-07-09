import numpy as np
import math

import random as rd
import time

class FitnessFun():
    def __init__(self):
        self.count=0

    def chooseFunction(self,X,func:str):
        if func=='sphere':
            return self.cal(X=X)
        elif func=='schwefel':
            return self.Schwefel(X=X)
        elif func=='step':
            return self.Step(X=X)
        elif func=='ackley':
            return self.Ackley(X=X)
        elif func=='rosenbrock':
            return self.Rosenbrock(X=X)
        elif func=='griewank':
            return self.Griewank(X=X)
        elif func=='ridge':
            return self.Ridge(X=X)
        elif func=='deceptive':
            return self.Deceptive(X=X)
        elif func=='whitley':
            return self.Whitley(X=X)
        elif func=='moddouble':
            return self.Moddouble(X=X)
        elif func=='quartic':
            return self.Quartic(X=X)
        elif func=='rastrigin':
            return self.Rastrigin(X=X)



    def cal(self,X):
        # 根据函数要求，每个向量中的各个元素进行平方并累加可得函数值
        Vars=X.copy()
        for index in range(Vars.shape[1]):
            # print(index)
            # x=Vars[:,[index]]*Vars[:,[index]]
            Vars[:, [index]] *= Vars[:, [index]]  # 进行平方
        # print(Vars)
        Y = 0  # Result
        for index in range(Vars.shape[1]):
            Y = Y + Vars[:, [index]]  # 进行累加

        #模拟复杂程序

        self.count+=X.shape[0]
        #time.sleep(Vars.shape[0]*0.005)
        return Y

    def Schwefel(self,X):
        Vars=X.copy()
        f=[]
        for index in range(Vars.shape[0]):
            sum=0
            for j in range(Vars.shape[1]):
                sum+=Vars[index,j]*math.sin(math.sqrt(abs(Vars[index,j])))
            f.append(sum)
        for index in range(Vars.shape[0]):
            f[index]=418.9829*Vars.shape[1]-f[index]


        self.count+=X.shape[0]
        return np.array(f).reshape(X.shape[0],1)

    def Step(self,X):
        Vars=X.copy()
        temp=np.array([])
        for index in range(Vars.shape[1]):
            t=(Vars[:,index]+0.5)*(Vars[:,index]+0.5)
            if temp.size==0:
                temp=t
            else:
                temp+=t

        self.count += X.shape[0]
        return temp.reshape(X.shape[0],1)

    def Ackley(self, X:np):
        """"""
        Vars=X.copy()
        first = []
        second = []
        for row in range(Vars.shape[0]):
            firstSum=0.0
            secondSum=0.0
            for index in range(Vars.shape[1]):
                firstSum+=Vars[row,index] ** 2.0
                secondSum+=math.cos(2.0 * math.pi * Vars[row,index])
            first.append(firstSum)
            second.append(secondSum)
        first=np.array(first).reshape(X.shape[0],1)
        second=np.array(second).reshape(X.shape[0],1)
        n = float(Vars.shape[1])
        f=[]
        for row in range(Vars.shape[0]):
            f.append(-20.0 * math.exp(-0.2 * math.sqrt(first[row] / n)) - math.exp(second[row] / n) + 20 + math.e)
        self.count += X.shape[0]
        return np.array(f).reshape(X.shape[0],1)

    def Rosenbrock(self,X:np):
        """F8 Rosenbrock's saddle
        	multimodal, asymmetric, inseparable"""
        Vars=X.copy()
        fitness = 0
        for i in range(Vars.shape[1] - 1):
            fitness += 100 * ((Vars[:,i] ** 2) - Vars[:,i + 1]) ** 2 + (1 - Vars[:,i]) ** 2
        self.count += X.shape[0]
        return np.array(fitness).reshape(X.shape[0],1)

    def Griewank(self, X):
        """F6 Griewank's function
        multimodal, symmetric, inseparable"""
        Vars=X.copy()
        part1 = 0
        for i in range(Vars.shape[1]):
            part1 += Vars[:,i] ** 2
            part2 = 1
        for row in range(X.shape[0]):
            for i in range(Vars.shape[1]):
                part2 *= math.cos(float(Vars[row,i]) / math.sqrt(i + 1))
        self.count += X.shape[0]
        return np.array(1 + (part1 / 4000.0) - part2 ).reshape(X.shape[0],1)

    def Ridge(self, X:np):
        """F2 Ridge's function
        unimodal, symmetric, inseparable"""
        Vars=X.copy()
        fitness = 0
        for i in range(Vars.shape[1]):
            temp = 0
            for j in range(i+1):
                temp += Vars[:,j]
            fitness += temp ** 2
        self.count += X.shape[0]
        return np.array(fitness).reshape(X.shape[0],1)



    def Deceptive(self, X:np):
        Vars=X.copy()
        deceptiveness = 0.20  # This is actually more like the inverse of deceptiveness since smaller = more deceptive.
        best_fitness = 1.0
        deceptive_best = 0.7


        dimensions = Vars.shape[1]
        f=[]
        for row in range(Vars.shape[0]):
            fitness = 0
            for i in range(dimensions):
                if Vars[row,i] < deceptiveness:
                    # Then fitness value is on a negative slope with a y
                    # intercept at 1
                    fitness += Vars[row,i] * (-1.0 / deceptiveness) \
                               + best_fitness
                else:
                    # Otherwise, the fitness value is on a positive slope
                    # with an x intercept at deceptiveness
                    fitness += (Vars[row,i] - deceptiveness) * \
                               (deceptive_best / (1.0 - deceptiveness))
            f.append(fitness)

        self.count += X.shape[0]
        return np.array(f).reshape(X.shape[0],1) / float(dimensions)

    def Whitley(self, X:np):
        Vars=X.copy()
        """F9 Whitley's function
        multimodal, asymmetric, inseparable
        http://www.it.lut.fi/ip/evo/functions/node13.html """
        fitness = 0
        limit = Vars.shape[1]
        for i in range(limit):
            for j in range(limit):
                temp = 100 * ((Vars[:,i] ** 2) - Vars[:,j]) + \
                       (1 - Vars[:,j]) ** 2
                temp=np.array(temp).reshape(Vars.shape[0],1)
                a=[]
                for t in temp:
                   a.append(math.cos(t))
                a=np.array(a).reshape(Vars.shape[0],1)
                fitness += ((temp ** 2) / 4000.0) - a + 1
        self.count += X.shape[0]
        return fitness

    def Moddouble(self, X:np):
        Vars=X.copy()
        """F4 Modified double sum
        unimodal, asymmetric, inseparable"""
        fitness = 0
        for i in range(Vars.shape[1]):
            for j in range(i + 1):
                fitness += (Vars[:,j] - (j + 1)) ** 2
        self.count += X.shape[0]
        return np.array(fitness).reshape(X.shape[0],1)

    def Quartic(self, X:np):
        """"""
        Vars=X.copy()
        total = 0.0
        for i in range(Vars.shape[1]):
            total += (i + 1.0) * Vars[:,i] ** 4.0
        for row in range(Vars.shape[0]):
            total[row]+=rd.random()
        self.count += X.shape[0]
        return np.array(total).reshape(X.shape[0],1)

    def Rastrigin(self, X:np):
        Vars=X.copy()
        """F5 Rastrigin's function
        multimodal, symmetric, separable"""
        f=[]
        for row in range(Vars.shape[0]):
            fitness = 10 * Vars.shape[1]
            for i in range(Vars.shape[1]):
                fitness += Vars[row,i] ** 2 - (10 * math.cos(2 * math.pi * Vars[row,i]))
            f.append(fitness)
        self.count += X.shape[0]
        return np.array(f).reshape(X.shape[0],1)