import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    
    if len(sys.argv) < 2:
        print("Usage: python kmeans_base.py <percentage>")
        sys.exit(1)

    percentage = float(sys.argv[1])

    # Load the dataset
    dataset = pd.read_csv("/afs/ece.cmu.edu/usr/zhuoqili/Private/project/dataset/data/f1_data/processed/final_processed_adjusted.csv")

    # Split based on percentage
    split_len = int(len(dataset) * percentage)
    dataset = dataset[:split_len]

    print(f"Dataset: {dataset.shape[0]} points, {dataset.shape[1]} dimensions")

    init_centroids = dataset[:5].copy()

    kmeans = KMeans(n_clusters=5, max_iter=500, n_init=1, tol=0, init=init_centroids, algorithm='lloyd')

    start = time.time()
    kmeans.fit(dataset)
    end = time.time()
    duration_ms = (end - start) * 1000
    duration_ms_per_iter = duration_ms / kmeans.n_iter_
    print(f"total time: {duration_ms:.2f} ms")
    print(f"iterations: {kmeans.n_iter_}")
    print(f"time per iteration: {duration_ms_per_iter:.2f} ms")
    print(f"inertia: {kmeans.inertia_}")

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    feature_names = list(dataset.columns)
    k = centers.shape[0]

    counts = np.bincount(labels, minlength=k)
    pct = counts / counts.sum() * 100

    speed_idx = feature_names.index("Speed")
    throttle_idx = feature_names.index("Throttle")
    brake_idx = feature_names.index("Brake")

    def style_name(c):
        s, t, b = c[speed_idx], c[throttle_idx], c[brake_idx]
        if b > 0.3:
            return "Heavy braking"
        if t > 0.3 and s > 0.3:
            return "Full throttle / high speed"
        if s < -0.3 and t < 0:
            return "Slow corner"
        if t > 0.0 and abs(s) < 0.5:
            return "Mid-speed cruise"
        return "Transition"

    print("\n--- Driving style distribution ---")
    for i in range(k):
        name = style_name(centers[i])
        print(f"Cluster {i} [{name}]: {pct[i]:.2f}%  "
              f"(Speed={centers[i, speed_idx]:.2f}, "
              f"Throttle={centers[i, throttle_idx]:.2f}, "
              f"Brake={centers[i, brake_idx]:.2f})")

    x_idx = feature_names.index("X")
    y_idx = feature_names.index("Y")
    xs = dataset.iloc[:, x_idx].values
    ys = dataset.iloc[:, y_idx].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter = ax1.scatter(xs, ys, c=labels, cmap="tab10", s=2)
    ax1.set_title("Track map colored by driving-style cluster")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(*scatter.legend_elements(), title="Cluster", loc="best")

    bar_labels = [f"C{i}\n{style_name(centers[i])}" for i in range(k)]
    ax2.bar(bar_labels, pct, color=[plt.cm.tab10(i) for i in range(k)])
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Driving-style distribution")
    for i, v in enumerate(pct):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    out_path = "/afs/ece.cmu.edu/usr/zhuoqili/Private/project/external_baseline/kmeans_styles.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")

if __name__ == "__main__":
    main()
