from mlrGAtsne.main import main
from mlrGAtsne.Configuration import Configuration
from mlrGAtsne.draw import benchmark
from mlrGAtsne.draw import draw

experiment_times=1

#func=['step','sphere','schwefel','ackley','rosenbrock','griewank','ridge','deceptive','whitley','moddouble','quartic','rastrigin']
#mode=['sga','mlr_sga','lle_mlr_sga','tsne_mlr_sga','lle_ap_mlr_sga','tsne_ap_mlr_sga','isomap_mlr_sga','isomap_ap_mlr_sga']
func=['sphere']
mode=['sga','lle_mlr_sga','lle_ap_mlr_sga']
for index in range(len(func)):
    for j in range(len(mode)):
        for k in range(experiment_times):
            if func[index]=='step'or func[index]=='sphere':
                con=Configuration(function=func[index],mode=mode[j])
            elif func[index]=='schwefel':
                con=Configuration(lb=-500,ub=500,function=func[index],mode=mode[j])
            elif func[index]=='ackley':
                con=Configuration(lb=-32,ub=32,function=func[index],mode=mode[j])
            elif func[index]=='rosenbrock':
                con=Configuration(lb=-2.048,ub=2.048,function=func[index],mode=mode[j])
            elif func[index]=='griewank':
                con=Configuration(lb=-512,ub=512,function=func[index],mode=mode[j])
            elif func[index]=='ridge':
                con=Configuration(lb=-64,ub=64,function=func[index],mode=mode[j])
            elif func[index]=='deceptive':
                con = Configuration(lb=0, ub=1, function=func[index], mode=mode[j])
            elif func[index]=='whitley' or func[index]=='moddouble':
                con = Configuration(lb=-10.24, ub=10.24, function=func[index], mode=mode[j])
            elif func[index]=='quartic':
                con = Configuration(lb=-1.28, ub=1.28, function=func[index], mode=mode[j])
            elif func[index]=='rastrigin':
                con = Configuration(lb=-5.12, ub=5.12, function=func[index], mode=mode[j])

            main(con)
            benchmark(con)
    draw(con,mode)
