import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def display(list_points, list_closest, name_file):
    points = list_points.collect()
    closest = list_closest.collect()

    pca = PCA(n_components=2)
    data = pca.fit_transform(points)
    plt.scatter(data[:, 0], data[:, 1], c=closest, alpha=0.5)

    plt.legend(loc='upper left')
    plt.title(label=name_file)
    plt.show()

    return
