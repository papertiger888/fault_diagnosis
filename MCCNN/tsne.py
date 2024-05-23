import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 生成模拟数据
n_samples = 5000
n_features = 100
n_clusters = 10 # 类别数改为10
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# 模拟不同阶段的数据变化
# 阶段1: 原始数据
X_stage1 = X + np.random.normal(0, 1.5, X.shape)
# 阶段2: 特征变得稍微区分一些
X_stage2 = X + np.random.normal(0, 1.2, X.shape)
# 阶段3: 特征变得更加区分
X_stage3 = X + np.random.normal(0, 0.8, X.shape)
# 阶段4: 特征变得非常区分
X_stage4 = X + np.random.normal(0, 0.5, X.shape)
# 阶段5: 特征高度区分
X_stage5 = X + np.random.normal(0, 0.2, X.shape)

# 进行t-SNE
tsne = TSNE(n_components=2,perplexity=30, random_state=42)
X_tsne_stage1 = tsne.fit_transform(X_stage1)
X_tsne_stage2 = tsne.fit_transform(X_stage2)
X_tsne_stage3 = tsne.fit_transform(X_stage3)
X_tsne_stage4 = tsne.fit_transform(X_stage4)
X_tsne_stage5 = tsne.fit_transform(X_stage5)

# 可视化
fig, axs = plt.subplots(1, 5, figsize=(25, 5))

for i, (X_tsne, title) in enumerate(zip([X_tsne_stage1, X_tsne_stage2, X_tsne_stage3, X_tsne_stage4, X_tsne_stage5],
                                        ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'])):
    axs[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=1) # 使用tab10颜色图
    axs[i].set_title(title)
    axs[i].axis('off')

plt.show()