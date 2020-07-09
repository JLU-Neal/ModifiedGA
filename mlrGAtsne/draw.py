from matplotlib import pyplot as plt
import numpy as np
import csv
from mlrGAtsne.Configuration import Configuration
#mode=['sga','mlr_sga','lle_mlr_sga','tsne_mlr_sga','lle_ap_mlr_sga','tsne_ap_mlr_sga']
"""



"""
def benchmark(con:Configuration):

    X0=[]

    Y0=[]
    with open(con.function+' based on '+con.mode+".csv","r") as csvfile:
        reader=csv.reader(csvfile)
        dict={}
        best_fitness=[]
        for row in reader:
            X0.append(int(row[0]))
            Y0.append(int(row[1]))
            dict[int(row[0])]=0
        for index in range(len(X0)):
            dict[X0[index]]+=Y0[index]
            if X0[index]==con.MAXGEN-1:
                best_fitness.append(Y0[index])

        a=[]
        b=[]
        for index in range(len(dict)):
            a.append(index)
            b.append(dict[index]/10)
        with open("figure_data of "+con.function+' based on '+con.mode+".csv","w",newline='')as csvfileW:
            writer=csv.writer(csvfileW)
            for index in range(len(a)):
                writer.writerow([a[index],b[index]])

        #benchmark
        best_fitness=np.array(best_fitness)
        best=best_fitness.min()
        average=best_fitness.sum()/best_fitness.size
        std=best_fitness.std()
        with open("benchmark.csv", "a", newline='') as csvfileB:
            writer = csv.writer(csvfileB)
            writer.writerow([con.function+' based on '+con.mode, 'best fitness',best])
            writer.writerow(['', 'average fitness', average])
            writer.writerow(['', '标准差', std])
            writer.writerow(['', 'fitness calling counts', ''])
            writer.writerow(['', 'average running time', ''])




def draw(con:Configuration,mode):
    color=['r','g','b','brown','gold','grey','m','k']
    fig=plt.figure()
    for index in range(len(mode)):
        X1=[]

        Y1=[]
        with open("figure_data of "+con.function+' based on '+mode[index]+".csv","r") as csvfile:
            reader=csv.reader(csvfile)
            for row in reader:
                X1.append(int(row[0]))
                Y1.append(float(row[1]))
        plt.plot(X1, Y1, label=mode[index], c=color[index])
    plt.title('figure of '+con.function)
    plt.legend()
    plt.savefig('figure of '+con.function+'.png')




X2 = []

Y2 = []
with open("MLR_LLE.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X2.append(int(row[0]))
        Y2.append(float(row[1]))

X3=[]

Y3=[]
with open("MLR_LLE_AP.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X3.append(int(row[0]))
        Y3.append(float(row[1]))

X4=[]

Y4=[]
with open("traditionalGA.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X4.append(int(row[0]))
        Y4.append(float(row[1]))

X5=[]

Y5=[]
with open("MLR_TSNE.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X5.append(int(row[0]))
        Y5.append(float(row[1]))

X6=[]

Y6=[]
with open("MLR_TSNE_AP.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X6.append(int(row[0]))
        Y6.append(float(row[1]))

X7=[]

Y7=[]
with open("traditionalGA(schwefel).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X7.append(int(row[0]))
        Y7.append(float(row[1]))

X8=[]

Y8=[]
with open("MLR_LLE(schwefel).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X8.append(int(row[0]))
        Y8.append(float(row[1]))


X9=[]

Y9=[]
with open("MLR_LLE_AP(schwefel).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X9.append(int(row[0]))
        Y9.append(float(row[1]))


X10=[]

Y10=[]
with open("MLR_ISOMAP_AP(schwefel).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X10.append(int(row[0]))
        Y10.append(float(row[1]))
X11=[]

Y11=[]
with open("traditionalGA(step).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X11.append(int(row[0]))
        Y11.append(float(row[1]))

X12=[]

Y12=[]
with open("MLR(step).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X12.append(int(row[0]))
        Y12.append(float(row[1]))

X13=[]

Y13=[]
with open("MLR_LLE(step).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X13.append(int(row[0]))
        Y13.append(float(row[1]))

X14=[]

Y14=[]
with open("MLR_LLE_AP(step).csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        X14.append(int(row[0]))
        Y14.append(float(row[1]))


X15=[]

Y15=[]
with open("MLR_TSNE_AP(step).csv","r") as csvfile:

    reader=csv.reader(csvfile)
    for row in reader:
        X15.append(int(row[0]))
        Y15.append(float(row[1]))


X16=[]

Y16=[]
with open("MLR_TSNE(step).csv","r") as csvfile:

    reader=csv.reader(csvfile)
    for row in reader:
        X16.append(int(row[0]))
        Y16.append(float(row[1]))

#fig=plt.figure()
#plt.plot(a,b,label='TEST')
#plt.plot(X1,Y1,label='MLR',c='r')
#plt.plot(X2,Y2,label='MLR_LLE',c='g')
#plt.plot(X3,Y3,label='MLR_LLE_AP',c='b')
#plt.plot(X4,Y4,label='traditional GA',c='brown')
#plt.plot(X5,Y5,label='MLR_TSNE',c='gold')
#plt.plot(X6,Y6,label='MLR_TSNE_AP',c='grey')
#plt.plot(X7,Y7,label='traditionalGA(Schwefel)',c='orange')
#plt.plot(X8,Y8,label='MLR_LLE(schwefel)',c='red')
#plt.plot(X9,Y9,label='MLR_LLE_AP(schwefel)',c='green')
#plt.plot(X10,Y10,label='MLR_ISOMAP_AP(schwefel)',c='green')
#plt.plot(X11,Y11,label='traditionalGA(step)',c='orange')
#plt.plot(X12,Y12,label='MLR(step)',c='r')
#plt.plot(X13,Y13,label='MLR_LLE(step)',c='g')
#plt.plot(X14,Y14,label='MLR_LLE_AP(step)',c='grey')
#plt.plot(X15,Y15,label='MLR_TSNE_AP(step)',c='brown')
#plt.plot(X16,Y16,label='MLR_TSNE(step)',c='b')





#plt.legend()
#plt.show(fig)