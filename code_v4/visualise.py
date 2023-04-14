import argparse
import os
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('iteration', type=int)

    return parser.parse_args()

def main():
    args = parse_args()

    experiment_dir = os.path.join('experiments', args.experiment_name)
    result_path = os.path.join(experiment_dir, 'results', f'iteration_{args.iteration}.json')
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(result_path, 'r') as f:
        data = json.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)

    names = []
    name_txt = ''
    for i, result in enumerate(data['all_results']):
        if i % 10 == 0:
            name_txt += '\n'
        name = result['video_name']
        names.append(name)
        name_txt += f'{name} '
    name_txt = name_txt[1:]

    while True:
        command = input('exit/names/config/{video_name}: ')

        if command == 'exit':
            break

        elif command == 'names':
            for i,name in enumerate(names):
                print(i+1, name)
            continue

        elif command == 'config':
            for key in config:
                print(f'{key}: {config[key]}')
            print()

        for result in data['all_results']:
            if result['video_name'] == command:
                progress = result['progress']
                predicted = result['predicted_progress']
                # get frames
                plt.plot(progress, label='progress')
                plt.plot(predicted, label='predicted')
                plt.title(f'{command}')
                plt.legend(loc='best')
                plt.xlabel('Frame')
                plt.ylabel('Progress (%)')
                plt.savefig(os.path.join(experiment_dir, 'tmp.png'))
                plt.clf()
        



if __name__ == '__main__':
    main()