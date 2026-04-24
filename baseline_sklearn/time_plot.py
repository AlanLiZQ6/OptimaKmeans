import matplotlib.pyplot as plt
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "results_py.csv"))

plt.plot(df["percentage"]*3800046, df["time_per_iter"], marker="o")
plt.xlabel("Number of Data Points")
plt.ylabel("Time (ms)")
plt.title("KMeans Execution Time vs Data Size (Sklearn Kmeans lloyd Baseline )")
plt.grid(True)
plt.savefig(os.path.join(current_dir, "time_plot_py.png"))
