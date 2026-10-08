
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = "plots/"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("Iris.csv")
df = df.drop(columns="Id")
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

print("=== First 5 rows ===");          print(df.head())
print("\n=== Shape ===", df.shape)
print("\n=== Info ===");                 df.info()
print("\n=== Missing values ===");       print(df.isnull().sum())
print("\n=== Duplicate rows ===", df.duplicated().sum())
print("\n=== Class counts ===");         print(df["Species"].value_counts())

print("\n=== Summary statistics ===")
print(df[features].describe().round(3))
print("\nMedian:\n", df[features].median())
print("\nVariance:\n", df[features].var().round(3))
print("\nSkewness:\n", df[features].skew().round(3))
print("\n=== Mean by species ===");      print(df.groupby("Species")[features].mean().round(3))
print("\n=== Std by species ===");       print(df.groupby("Species")[features].std().round(3))
print("\n=== Correlation matrix ===");   print(df[features].corr().round(3))

# 1. Histograms
df[features].hist(bins=15, figsize=(10, 7), edgecolor="black", color="steelblue")
plt.suptitle("Feature Distributions"); plt.tight_layout()
plt.savefig(OUT + "01_histograms.png", dpi=150); plt.close()

# 2. Boxplots by species
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, f in zip(axes.ravel(), features):
    sns.boxplot(data=df, x="Species", y=f, hue="Species", ax=ax, legend=False)
    ax.set_title(f"{f} by Species")
plt.tight_layout(); plt.savefig(OUT + "02_boxplots.png", dpi=150); plt.close()

# 3. Violin plots
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, f in zip(axes.ravel(), features):
    sns.violinplot(data=df, x="Species", y=f, hue="Species", ax=ax, legend=False)
    ax.set_title(f"{f} Distribution")
plt.tight_layout(); plt.savefig(OUT + "03_violins.png", dpi=150); plt.close()

# 4. Pairplot
sns.pairplot(df, hue="Species", diag_kind="kde", height=2.2)
plt.savefig(OUT + "04_pairplot.png", dpi=130); plt.close()

# 5. Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", ax=axes[0])
axes[0].set_title("Sepal Length vs Width")
sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species", ax=axes[1])
axes[1].set_title("Petal Length vs Width")
plt.tight_layout(); plt.savefig(OUT + "05_scatter.png", dpi=150); plt.close()

# 6. Correlation heatmap
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig(OUT + "06_heatmap.png", dpi=150); plt.close()

# 7. Species count
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Species", hue="Species", legend=False)
plt.title("Samples per Species"); plt.tight_layout()
plt.savefig(OUT + "07_countplot.png", dpi=150); plt.close()

print("\nPlots saved in the 'plots' folder.")
