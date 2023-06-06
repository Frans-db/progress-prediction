import argparse
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    # wandb
    parser.add_argument("--wandb_project", type=str, default="mscfransdeboer")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_disable", action="store_true")
    # data
    parser.add_argument("--dataset", type=str, default="cholec80")
    parser.add_argument("--data_dir", type=str, default="features/i3d_embeddings")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--bboxes", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--indices", action="store_true")
    parser.add_argument("--indices_normalizer", type=float, default=1.0)
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument(
        "--rsd_type", type=str, default="none", choices=["none", "minutes", "seconds"]
    )
    parser.add_argument("--rsd_normalizer", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--no_resize", action="store_true")
    parser.add_argument("--train_split", type=str, default="train.txt")
    parser.add_argument("--test_split", type=str, default="test.txt")
    # training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10000)
    # network
    parser.add_argument(
        "--network",
        type=str,
        default="progressnet",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["vgg16", "resnet18", "resnet152"],
    )
    parser.add_argument("--load_backbone", type=str, default=None)
    # network parameters
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--dropout_chance", type=float, default=0.0)
    parser.add_argument("--pooling_layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--roi_size", type=int, default=4)
    # network loading
    parser.add_argument("--load_experiment", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
    # optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument(
        "--loss", type=str, default="l2", choices=["l2", "l1", "smooth_l1"]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    # scheduler
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay_every", type=int, default=503 * 30)
    # logging
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--test_every", type=int, default=1000)

    parser.add_argument("--print_only", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--embed_batch_size", type=int, default=10)
    parser.add_argument("--embed_dir", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--save', action='store_true')

    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def wandb_init(args):
    no_wandb = args.wandb_disable or args.print_only or args.eval or args.embed or args.save
    if no_wandb:
        return

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config={
            # experiment
            'seed': args.seed,
            # data
            'dataset': args.dataset,
            'data_dir': args.data_dir,
            'flat': args.flat,
            'bboxes': args.bboxes,
            'shuffle': args.shuffle,
            'indices': args.indices,
            'indices_normalizer': args.indices_normalizer,
            'subsample': args.subsample,
            'rsd_type': args.rsd_type,
            'rsd_normalizer': args.rsd_normalizer,
            'fps': args.fps,
            'no_resize': args.no_resize,
            'train_split': args.train_split,
            'test_split': args.test_split,
            # training
            'batch_size': args.batch_size,
            'iterations': args.iterations,
            # network
            'network': args.network,
            'backbone': args.backbone,
            'load_backbone': args.load_backbone,
            # network parameters
            'feature_dim': args.feature_dim,
            'embed_dim': args.embed_dim,
            'dropout_chance': args.dropout_chance,
            'pooling_layers': args.pooling_layers,
            'roi_size': args.roi_size,
            # network loading
            'load_experiment': args.load_experiment,
            'load_iteration': args.load_iteration,
            # optimizer
            'optimizer': args.optimizer,
            'loss': args.loss,
            'momentum': args.momentum,
            'betas': (args.beta1, args.beta2),
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            # scheduler
            'lr_deday': args.lr_decay,
            'lr_decay_every': args.lr_decay_every,
        },
    )