import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import mnist
from time import time
for perplexity in [10, 50, 100]:
    print(f"\nExecutando t-SNE no MNIST com perplexidade={perplexity}...")
    tsne_experiment = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    start_time_exp = time()
    X_tsne_exp = tsne_experiment.fit_transform(X_train_flat[:2000])
    end_time_exp = time()


    plt.figure(figsize=(10, 8))
    scatter_exp = plt.scatter(X_tsne_exp[:, 0], X_tsne_exp[:, 1], c=y_train[:2000], cmap="tab10", s=10)
    plt.colorbar(scatter_exp)
    plt.title(f"MNIST (t-SNE com perplexidade={perplexity})")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.show()

    print(f"Tempo de execução com perplexidade={perplexity}: {end_time_exp - start_time_exp:.2f} segundos")