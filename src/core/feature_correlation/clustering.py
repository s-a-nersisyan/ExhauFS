import scipy.cluster.hierarchy as spc


def features_clusters(df, ann, features, tresh=0.5):
    corr = df[features].corr().values

    pdist = spc.distance.pdist(corr)
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, tresh * pdist.max(), 'distance')

    return idx
