import argparse

def parse_arguments():
    # TODO: Clean up arguments
    # perhaps split into segments?

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    # directories
    parser.add_argument('--experiment_directory', type=str, default='experiments')
    parser.add_argument('--model_directory', type=str, default='models')
    parser.add_argument('--log_directory', type=str, default='logs')
    parser.add_argument('--figures_directory', type=str, default='figures')
    # figures
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--figure_every', type=int, default=25)
    # dataset
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets')
    parser.add_argument('--dataset', type=str, default='ucf24')
    parser.add_argument('--data_type', type=str, default='rgb-images')
    parser.add_argument('--splitfile_dir', type=str, default='splitfiles')
    parser.add_argument('--annotation_file', type=str, default='pyannot.pkl')
    parser.add_argument('--train_split_file', type=str, default='trainlist01.txt')
    parser.add_argument('--test_split_file', type=str, default='testlist01.txt')
    # training
    parser.add_argument('--epochs', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--subsection_chance', type=float, default=0)
    parser.add_argument('--subsample_chance', type=float, default=0)
    parser.add_argument('--dropout_chance', type=float, default=0)
    parser.add_argument('--finetune', action='store_true')
    # progressnet
    parser.add_argument('--embed_size', type=int, default=2048)
    parser.add_argument('--model_type', type=str, default='progressnet')
    # model loading
    parser.add_argument('--model_name', type=str, default=None)
    # eval
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--relative', action='store_true')

    return parser.parse_args()