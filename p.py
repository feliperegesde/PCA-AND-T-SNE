import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
iris = load_iris()
iris_data = iris.data
iris_target = iris.target
target_names = iris.target_names


pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_data)
plt.figure(figsize=(10, 5))
sns.scatterplot(x=iris_pca[:, 0], y=iris_pca[:, 1], hue=iris_target, palette="viridis", s=100)
plt.title("PCA: Iris Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Classes", labels=target_names)
plt.show()
tsne = TSNE(n_components=2, random_state=42, n_iter=500, perplexity=30)
iris_tsne = tsne.fit_transform(iris_data)
plt.figure(figsize=(10, 5))
sns.scatterplot(x=iris_tsne[:, 0], y=iris_tsne[:, 1], hue=iris_target, palette="viridis", s=100)
plt.title("t-SNE: Iris Dataset (2D)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Classes", labels=target_names)
plt.show()
