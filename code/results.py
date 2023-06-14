results = {
    'ucf24': {
        'ResNet152': {
            'normal': 25.8,
            'segments': 25.8,
            'indices': 14.1,
        },
        'ResNet152-LSTM': {
            'normal': 24.4,
            'random': 17.1,
            'segments': 24.5,
            'indices': 18.0,
        },
        'UTE': {
            'normal': 24.7,
            'segments': 24.7,
            'indices': 14.1,
        },
        'ProgressNet': {
            'normal': 13.6,
            'random': 17.4,
            'segments': 26.6,
            'indices': 19.2,
        },
        'RSDNet': { # TODO: Waiting on results
            'normal': 0,
            'random': 0,
            'segments': 0,
            'indices': 0,
        },
        'Static 0.5': {
            'normal': 25.0,
            'random': 25.0,
            'segments': 25.0,
            'indices': 25.0,
        },
        'Random': {
            'normal': 33.3,
            'random': 33.3,
            'segments': 33.3,
            'indices': 33.3,
        },
        'Average Index': {
            'normal': 14.2,
            'random': 14.2,
            'segments': 14.2,
            'indices': 14.2,
        },
    },
    'cholec80 (sampled)': {
        'ResNet152': {
            'normal': 17.9,
            'segments': 17.9,
            'indices': 12.3,
        },
        'ResNet152-LSTM': { # TODO: Waiting
            'normal': 0,
            'random': 0,
            'segments': 0,
            'indices': 0
        },
        'UTE': { # TODO: Waiting
            'normal': 0,
            'segments': 0,
            'indices': 0,
        },
        'ProgressNet': { # TODO: Run 2
            'normal': 12.0,
            'random': 13.5,
            'segments': 0,
            'indices': 0,
        },
        'RSDNet': { # TODO: Waiting
            'normal': 0,
            'random': 0,
            'segments': 0,
            'indices': 0,
        },
        'Static 0.5': {
            'normal': 25.0,
            'random': 25.0,
            'segments': 25.0,
            'indices': 25.0,
        },
        'Random': {
            'normal': 33.3,
            'random': 33.3,
            'segments': 33.3,
            'indices': 33.3,
        },
        'Average Index': {
            'normal': 11.9,
            'random': 11.9,
            'segments': 11.9,
            'indices': 11.9,
        },
    },
    # vvvvvvvv
    'cholec80': {
        'ResNet152': {
            'normal': 17.9,
            'indices': 12.3,
        },
        'ResNet152-LSTM': {
            'normal': 0,
            'random': 0,
            'segments': 0,
            'indices': 0
        },
        'UTE': {
            'normal': 0,
            'indices': 0,
        },
        'RSDNet': {
            'normal': 0,
            'random': 0,
            'segments': 0,
            'indices': 0,
        },
        'Static 0.5': {
            'normal': 25.0,
            'random': 25.0,
            'segments': 25.0,
            'indices': 25.0,
        },
        'Random': {
            'normal': 33.3,
            'random': 33.3,
            'segments': 33.3,
            'indices': 33.3,
        },
        'Average Index': {
            'normal': 11.9,
            'random': 11.9,
            'segments': 11.9,
            'indices': 11.9,
        },
    }
}

def compare(dataset: str, mode1: str, mode2: str):
    dataset_results = results[dataset]

def main():
    pass

if __name__ == '__main__':
    main()