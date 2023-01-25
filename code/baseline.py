import matplotlib.pyplot as plt


# results for temporal embedding training of mlp
# final epoch of training
mse_results = {
    'default': 0.308134,
    'drop': 0.427060,
    'repeat': 0.830325,
    'shuffle': 1.200668,
}

labels, losses = zip(*mse_results.items())
plt.bar(labels, losses)
plt.title('Loss on different datasets (80 epochs)')
plt.xlabel('Dataset')
plt.ylabel('Final MSE Loss')
plt.show() 