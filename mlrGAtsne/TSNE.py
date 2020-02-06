from sklearn import manifold




class TSNE():
    def __init__(self,n_components):
        self.tsne=manifold.TSNE(n_components=n_components,init='pca')

    def transform(self,data):
        embeded_data=self.tsne.fit_transform(data)
        return embeded_data
