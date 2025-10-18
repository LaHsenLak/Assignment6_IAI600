import math
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib as mpl
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(path="housing.csv"):
    df = pd.read_csv(path)
    cols = ["longitude", "latitude", "median_income"]
    return df.dropna(subset=cols)[cols]

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_decision_boundaries(clusterer, X, resolution=1000,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

df = load_data()

# Scale features so income doesn't dominate distance
X = df[["longitude","latitude","median_income"]].to_numpy()

n = 7
ks = list(range(2, n + 1))
num_plots = len(ks)
cols = 3
rows = math.ceil(num_plots / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), squeeze=False)

dict_silh_score = dict()
silh_score = []

# Clusters, labels, 

# Plotting figures with predicting clusters
for i, k in enumerate(ks):
    ax = axes[i // cols, i % cols]

    km = KMeans(n_clusters=k, algorithm="lloyd", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silh_score.append(score)
    dict_silh_score[k] = labels

    silh = silhouette_samples(X, labels)

    padding = len(X) // 30
    pos = padding
    ticks = []

    for c in range(k):
        coeffs = np.sort(silh[labels == c])
        y = np.arange(pos, pos + len(coeffs))
        color = mpl.cm.Spectral(c / k)
        ax.fill_betweenx(y, 0, coeffs, facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(list(range(k))))
    ax.axvline(x=score, color="red", linestyle="--")
    ax.set_title(f"k={k}", fontsize=12)
    ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")

for j in range(num_plots, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.show()

# Silhouette score
plt.figure(figsize=(8, 3))
plt.plot(ks, silh_score, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.show()
