def vis(df):
    from sklearn.decomposition import PCA
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    pca = PCA(n_components=3)

    pca.fit(df.values)
    x_pca = pd.DataFrame(pca.transform(df.values), columns=(["col1","col2", "col3"]))
    kmeans = KMeans(n_clusters=3)
    pre_kmeans=kmeans.fit_predict(x_pca)

    fig=plt.figure(figsize=(20,15))
    plot=plt.subplot(111,projection='3d',label="bla")
    plot.set_title("The Plot of KMeans Clustering",fontsize=30)
    plot.scatter(x_pca['col1'],x_pca['col2'],x_pca['col3'],s=150,c=pre_kmeans,marker='o', cmap='viridis',zorder=10)
    plt.savefig('vis.png')