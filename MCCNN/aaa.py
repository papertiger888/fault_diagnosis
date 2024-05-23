import matplotlib.pyplot as plt
import numpy as np

num_src = 3
num_tgt = 4
num_samples = 200

fig, axs = plt.subplots(num_src, num_tgt, figsize=(12, 9))

for i in range(num_src):
    for j in range(num_tgt):
        alphas = np.random.beta(0.5, 0.5, size=num_samples)
        counts, bins, _ = axs[i, j].hist(alphas, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axs[i, j].set_title(f'Source {i+1} to Target {j+1}')
        axs[i, j].set_xlabel('Alpha values')
        axs[i, j].set_ylabel('Sample count')

plt.suptitle('Alpha Distributions across Source and Target Domains', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()