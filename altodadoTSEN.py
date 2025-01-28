import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import mnist
from time import time

plt.style.use("ggplot")

(X_train, y_train), (_, _) = mnist.load_data()

X_train_flat = X_train.reshape(X_train.shape[0], -1)


tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
start_time = time()
X_tsne_mnist = tsne.fit_transform(X_train_flat[:2000])
end_time = time()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne_mnist[:, 0], X_tsne_mnist[:, 1], c=y_train[:2000], cmap="tab10", s=10)
plt.colorbar(scatter)
plt.title("MNIST (Redução de Dimensões com t-SNE)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()

print(f"Tempo de execução do t-SNE no MNIST: {end_time - start_time:.2f} segundos")





