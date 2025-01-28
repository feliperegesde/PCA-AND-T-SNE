import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


file_path = "C:/Users/User/Downloads/diabetes - diabetes.csv"  
diabetes_data = pd.read_csv(file_path)


X = diabetes_data.drop(columns=["Outcome"])  
y = diabetes_data["Outcome"]                


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
diabetes_pca = pca.fit_transform(X_scaled)


plt.figure(figsize=(10, 5))
sns.scatterplot(x=diabetes_pca[:, 0], y=diabetes_pca[:, 1], hue=y, palette="viridis", s=100)
plt.title("PCA: Diabetes Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Classes", labels=["No Diabetes (0)", "Diabetes (1)"])
plt.show()


tsne = TSNE(n_components=2, random_state=42, n_iter=500, perplexity=30)
diabetes_tsne = tsne.fit_transform(X_scaled)


plt.figure(figsize=(10, 5))
sns.scatterplot(x=diabetes_tsne[:, 0], y=diabetes_tsne[:, 1], hue=y, palette="viridis", s=100)
plt.title("t-SNE: Diabetes Dataset (2D)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Classes", labels=["No Diabetes (0)", "Diabetes (1)"])
plt.show()
