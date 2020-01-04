import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits, make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def PCAcurve(data):
    pca = PCA().fit(data)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

def plotPCA(data, targets=None, n_components=2):

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(data)

    print('Original Data Shape {}'.format(data.shape))
    print('Post PCA shape {}'.format(projected.shape))

    if targets is not None:
        plt.figure()
        plt.scatter(projected[:,0], projected[:,1], c=digits.target, 
            edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('inferno', 10))

        plt.xlabel('component1')
        plt.ylabel('component2')
        plt.colorbar()
        plt.show()

    return projected

def plotKMeans(data, n_clusters=10):
    X = data 
    k_means = KMeans(n_clusters=n_clusters, init='random', max_iter=300) 
    y_km = k_means.fit_predict(X)

    plt.figure()
    for i in range(10):
        plt.scatter(X[y_km == i, 0], X[y_km == i, 1],
        label='cluster {}'.format(i+1))
    plt.show()

# plt.clf()

def plotPCAvectors(data, n_components=2):
    pca = PCA(n_components=n)
    pca.fit(data)
    print(pca.components_)
    print(pca.explained_variance_)
    
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    
    plt.axis('equal')
    plt.show()

def plotDBSCAN(data):
    plt.figure()

    # centers = [[1, 1], [-1, -1], [1, -1]]
    # data, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
    #                             random_state=0)

    X = StandardScaler().fit_transform(data)

    # Compute DBSCAN
    db = DBSCAN(eps=0.2, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)

    plt.title('Estimated number of clusters (DBSCAN): %d' % n_clusters_)
    plt.show()

def plot_tSNE(data, targets=None, n_components=2):
    # t-SNE scales quadratically with data (is memory intensive)

    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    projected = tsne.fit_transform(data)

    if targets is not None:
        plt.figure()
        plt.scatter(projected[:,0], projected[:,1], c=digits.target, 
            edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('inferno', 10))

        plt.xlabel('component1')
        plt.ylabel('component2')
        plt.colorbar()
        plt.show()

if __name__=='__main__':
    digits = load_digits()

    # projected = plotPCA(digits.data, targets=digits.target, n_components=2)
    plot_tSNE(digits.data, targets=digits.target, n_components=2)

    # plotKMeans(projected, n_clusters=10)
    # plotDBSCAN(projected)

    
