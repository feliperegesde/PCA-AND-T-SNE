import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import mnist

tsne = TSNE(n_components=2, random_state=42, n_iter=500, perplexity=30)
diabetes_tsne = tsne.fit_transform(X_scaled)


plt.figure(figsize=(10, 5))
sns.scatterplot(x=diabetes_tsne[:, 0], y=diabetes_tsne[:, 1], hue=y, palette="viridis", s=100)
plt.title("t-SNE: Diabetes Dataset (2D)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Classes", labels=["Sem Diabetes (0)", "Com Diabetes (1)"])
plt.show()
