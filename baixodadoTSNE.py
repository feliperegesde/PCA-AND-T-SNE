import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
print("\nCarregando o dataset Iris...")
iris = load_iris()
X_iris = iris.data
y_iris = iris.target


print("Executando t-SNE no dataset Iris...")
tsne_iris = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
start_time_iris = time()
X_tsne_iris = tsne_iris.fit_transform(X_iris)
end_time_iris = time()


plt.figure(figsize=(10, 8))
scatter_iris = plt.scatter(X_tsne_iris[:, 0], X_tsne_iris[:, 1], c=y_iris, cmap="viridis", s=50)
plt.colorbar(scatter_iris)
plt.title("Iris (Redução de Dimensões com t-SNE)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()

print(f"Tempo de execução do t-SNE no Iris: {end_time_iris - start_time_iris:.2f} segundos")