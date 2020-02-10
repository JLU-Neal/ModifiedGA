

class FitnessFun():
    def cal(self,Vars):
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
        return Y
