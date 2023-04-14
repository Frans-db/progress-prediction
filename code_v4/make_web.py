import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import time
from tqdm import tqdm
from multiprocessing.pool import Pool

data_root = '/home/frans/Datasets/breakfast'
root = './experiments'
experiment_name = 'breakfast_progressnet'
result = 'iteration_25000.json'

experiment_dir = os.path.join(root, experiment_name)
config_path = os.path.join(experiment_dir, 'config.json')
results_path = os.path.join(experiment_dir, 'results', result)

tmp_dir = os.path.join(experiment_dir, 'tmp')
video_dir = os.path.join(experiment_dir, 'videos')

def create_vid(arg):
    result, i = arg
    video_name = result['video_name'].replace('.txt', '')
    progress = result['progress']
    predicted_progress = result['predicted_progress']

    frame_dir = os.path.join(data_root, 'rgb-images', video_name)
    frame_names = sorted(os.listdir(frame_dir))
    for j, frame_name in tqdm(enumerate(frame_names), leave=False, total=len(frame_names)):
        frame_path = os.path.join(frame_dir, frame_name)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(Image.open(frame_path).convert('RGB'))
        ax2.plot(progress[:j+1], label='progress')
        ax2.plot(predicted_progress[:j+1], label='predicted progress')
        ax2.set_xlim(0, len(progress))
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Progress (%)')
        ax2.legend(loc='best')
        plt.savefig(os.path.join(tmp_dir, f'{j:05d}.png'))
        
        plt.clf()
        plt.close()

    subprocess.call([
        'ffmpeg', '-framerate', '15', '-i', os.path.join(tmp_dir, '%05d.png'), '-r', '30', '-pix_fmt', 'yuv420p', os.path.join(video_dir, f'{video_name.replace("/", "_")}.mp4')
    ])
    for j, frame_name in enumerate(frame_names):
        os.remove(os.path.join(tmp_dir, f'{j:05d}.png'))       

with open(results_path) as f:
    data = json.load(f)

args = [(result,i) for (i,result) in enumerate(data['all_results'])]

os.mkdir(tmp_dir)
os.mkdir(video_dir)
for arg in tqdm(args):
    create_vid(arg)
os.rmdir(tmp_dir)
    