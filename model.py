

def model(df, show_plot = True):
    from sklearn.decomposition import PCA
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    pca = PCA(n_components=3)

    pca.fit(df.values)
    x_pca = pd.DataFrame(pca.transform(df.values), columns=(["col1","col2", "col3"]))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)

    pre_kmeans=kmeans.fit_predict(x_pca)

    x_pca["Clusters"]=pre_kmeans
    df["Clusters"]=pre_kmeans

    # Count the number of records in each cluster
    cluster_counts = df['Clusters'].value_counts()

    # Define the file name
    file_name = "k.txt"

    # Save the counts to the text file
    with open(file_name, "w") as file:
        for cluster, count in cluster_counts.items():
            file.write(f"Cluster {cluster}: {count} records\n")

    if show_plot:
        fig=plt.figure(figsize=(20,15))
        plot=plt.subplot(111,projection='3d',label="bla")
        plot.set_title("The Plot of KMeans Clustering",fontsize=30)
        plot.scatter(x_pca['col1'],x_pca['col2'],x_pca['col3'],s=150,c=pre_kmeans,marker='o', cmap='viridis',zorder=10)
        plt.show()