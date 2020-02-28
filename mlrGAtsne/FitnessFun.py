import numpy as np
import math
class FitnessFun():
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



        return np.array(f).reshape(X.shape[0],1)