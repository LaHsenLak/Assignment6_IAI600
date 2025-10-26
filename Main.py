import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.colors import ListedColormap

def _palette(n):
    base = plt.get_cmap("tab20").colors
    if n <= len(base):
        colors = base[:n]
    else:
        reps = int(np.ceil(n / len(base)))
        colors = (base * reps)[:n]
    return ListedColormap(colors)

def _label_to_colors(labels, noise_color=(0.75, 0.75, 0.75, 1.0)):
    labels = np.asarray(labels)
    uniq = sorted([u for u in np.unique(labels) if u != -1])
    cmap = _palette(len(uniq))
    color_map = {u: cmap(i) for i, u in enumerate(uniq)}
    out = np.empty((labels.size, 4), dtype=float)
    for i, lab in enumerate(labels):
        out[i] = noise_color if lab == -1 else color_map[lab]
    return out

# Load & scale data
def load_data(path="housing.csv"):
    df = pd.read_csv(path)
    cols = ["longitude", "latitude", "median_income"]
    df = df.dropna(subset=cols).copy()
    return df[cols]

df = load_data()
X_raw = df.values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)


ks = list(range(2, 8))
silh_scores = []
per_k_labels = {}

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, algorithm="lloyd", random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silh_scores.append(score)
    per_k_labels[k] = labels

best_k = ks[int(np.argmax(silh_scores))]
print(f"Best k by silhouette: {best_k:.0f} (score={max(silh_scores):.3f})")

plt.figure(figsize=(6,3))
plt.plot(ks, silh_scores, "o-")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs k")
plt.tight_layout()
plt.show()

best_labels = per_k_labels[best_k]
sample_silh = silhouette_samples(X, best_labels)
padding = len(X)//30
ticks, pos = [], 0
plt.figure(figsize=(6,4))
for c in range(best_k):
    coeffs = np.sort(sample_silh[best_labels == c])
    y = np.arange(pos, pos + len(coeffs))
    plt.fill_betweenx(y, 0, coeffs, alpha=0.7)
    ticks.append(pos + len(coeffs)//2)
    pos += len(coeffs) + padding
plt.axvline(x=sample_silh.mean(), color="red", linestyle="--")
plt.yticks(ticks, [f"Cluster {i}" for i in range(best_k)])
plt.xlabel("Silhouette coefficient")
plt.title(f"Silhouette plot (k={best_k})")
plt.tight_layout()
plt.show()


km_best = KMeans(n_clusters=best_k, n_init=10, algorithm="lloyd", random_state=42)
labels = km_best.fit_predict(X)

plt.figure(figsize=(6,5))
plt.scatter(df["longitude"], df["latitude"], c=labels, s=5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"KMeans clusters on map (k={best_k})")
plt.tight_layout()
plt.show()


centers_scaled = km_best.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

cluster_summary = (
    pd.DataFrame({
        "cluster": labels,
        "longitude": df["longitude"].values,
        "latitude": df["latitude"].values,
        "median_income": df["median_income"].values
    })
    .groupby("cluster")
    .agg(
        n=("cluster", "size"),
        lon_mean=("longitude", "mean"),
        lat_mean=("latitude", "mean"),
        income_mean=("median_income", "mean"),
        income_median=("median_income", "median"),
        income_min=("median_income", "min"),
        income_max=("median_income", "max"),
    )
    .sort_index()
)

centers_df = pd.DataFrame(
    centers_original, columns=["center_lon", "center_lat", "center_income"]
).assign(cluster=lambda d: np.arange(len(d))).set_index("cluster")

profile = cluster_summary.join(centers_df, how="left")
print("\nCluster profile (original units):")
print(profile)


eps_grid = [0.15, 0.20, 0.25, 0.30]
min_samples_grid = [5, 10, 20]

best_db = None
best_db_score = -1.0
best_db_labels = None

for eps in eps_grid:
    for ms in min_samples_grid:
        db = DBSCAN(eps=eps, min_samples=ms)
        lbl = db.fit_predict(X)

        non_noise = lbl != -1
        uniq = set(lbl[non_noise])
        if len(uniq) < 2:
            continue
        try:
            score = silhouette_score(X[non_noise], lbl[non_noise])
        except Exception:
            continue
        if score > best_db_score:
            best_db_score = score
            best_db = (eps, ms)
            best_db_labels = lbl


kmeans_sil = silhouette_score(X, labels)

if best_db_labels is None:
    print("DBSCAN failed to find usable clusters with this small grid.")
else:
    eps, ms = best_db
    n_noise = int(np.sum(best_db_labels == -1))
    n_points = len(best_db_labels)
    n_clusters = len(set(best_db_labels) - {-1})
    print(f"Best DBSCAN: eps={eps}, min_samples={ms}, silhouette={best_db_score:.3f}")
    print(f"DBSCAN clusters: {n_clusters}, noise fraction: {n_noise/n_points:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)

    db_colors = _label_to_colors(best_db_labels, noise_color=(0.8, 0.8, 0.8, 1.0))
    axes[0].scatter(df["longitude"], df["latitude"], c=db_colors, s=6, linewidths=0)
    axes[0].set_title(f"DBSCAN (eps={eps}, min_samples={ms})")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    km_colors = _label_to_colors(labels, noise_color=(0.6, 0.6, 0.6, 1.0))
    axes[1].scatter(df["longitude"], df["latitude"], c=km_colors, s=6, linewidths=0)
    axes[1].set_title(f"KMeans (k={best_k})")
    axes[1].set_xlabel("Longitude")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4.8, 3.2))
    names = [f"KMeans k={best_k}", f"DBSCAN eps={eps}\nmin_samples={ms}"]
    scores = [kmeans_sil, best_db_score]
    bars = plt.bar(names, scores)
    plt.ylabel("Silhouette score")
    plt.ylim(0, max(scores) * 1.15)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}",
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.show()
plt.plot(ks, silh_score, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.show()
