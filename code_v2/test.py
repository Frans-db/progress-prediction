import os
import matplotlib.pyplot as plt
import numpy as np

path = '/home/frans/Datasets/cholec80/rgb-images'
save_path = '/home/frans/Downloads/figure.png'
video_names = sorted(os.listdir(path))
lengths = []
bars = {}
for i in range(12):
    bars[i*10] = 0

for video_name in video_names:
    video_path = os.path.join(path, video_name)
    frames = sorted(os.listdir(video_path))
    length = len(frames) / 60
    lengths.append(length)
    rounded_length = round(length / 10) * 10

    bars[rounded_length] += 1

print(np.quantile(lengths, 0.25), np.quantile(lengths, 0.5), np.quantile(lengths, 0.75))
plt.bar([key for key in bars], [bars[key] for key in bars])
plt.savefig(save_path)




