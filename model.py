def model(df):
    import pandas as pd

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)

    pca.fit(df.values)
    x_pca = pd.DataFrame(pca.transform(df.values), columns=(["col1","col2", "col3"]))
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)

    pre_kmeans=kmeans.fit_predict(x_pca)

    df["Clusters"]=pre_kmeans

    # Count the number of records in each cluster
    cluster_counts = df['Clusters'].value_counts()

    # Save the counts to the "k.txt" file
    with open("k.txt", "w") as file:
        for cluster, count in cluster_counts.items():
            file.write(f"Cluster {cluster}: {count} records\n")