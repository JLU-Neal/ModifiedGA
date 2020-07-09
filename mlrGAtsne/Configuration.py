


class Configuration:
    def __init__(self,split=25,datasize=200,Dim=30,lb=-100,ub=100,lbin=1,ubin=1,function='step',mode='sga',NIND=50,MAXGEN=100):
        self.split=split
        self.datasize=datasize
        self.Dim=Dim
        self.lb=lb#lower boundary
        self.ub=ub#upper boundary
        self.lbin=lbin
        self.ubin=ubin
        self.function=function
        self.mode=mode
        self.NIND=NIND#种群规模
        self.MAXGEN=MAXGEN



