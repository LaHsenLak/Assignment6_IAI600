import math
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib as mpl
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(path="housing.csv"):
    df = pd.read_csv(path)
    cols = ["longitude", "latitude", "median_income"]
    return df.dropna(subset=cols)[cols]

df = load_data()

# Scale features so income doesn't dominate distance
X = StandardScaler().fit_transform(df[["longitude","latitude","median_income"]].to_numpy())

n = 7
ks = list(range(2, n + 1))
num_plots = len(ks)
cols = 3
rows = math.ceil(num_plots / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), squeeze=False)

dict_silh_score = []

for i, k in enumerate(ks):
    ax = axes[i // cols, i % cols]

    km = KMeans(n_clusters=k, algorithm="lloyd", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    dict_silh_score.append(score)

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
    ax.set_xlim(-0.2, 1.0)  # typical silhouette range
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")

# Hide any unused axes
for j in range(num_plots, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(ks, dict_silh_score, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.show()
