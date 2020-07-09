from sklearn import manifold
from mlrGAtsne.Configuration import Configuration



class TSNE():
    def __init__(self,n_components,con:Configuration):
        if con.mode=='lle_mlr_sga' or con.mode=='lle_ap_mlr_sga':
            self.tsne = manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense')
        elif con.mode=='tsne_mlr_sga' or con.mode=='tsne_ap_mlr_sga':
            self.tsne=manifold.TSNE(n_components=n_components,init='pca',method='barnes_hut',angle=0.2,n_iter=1000)
        elif con.mode=='isomap_mlr_sga' or con.mode=='isomap_ap_mlr_sga':
            self.tsne=manifold.Isomap(n_components=n_components)

    def transform(self,data):
        embeded_data=self.tsne.fit_transform(data)
        return embeded_data
