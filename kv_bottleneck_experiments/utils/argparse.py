import argparse
import json
import os
import uuid


class ArgumentParserWrapper(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--plot_pair_utilization', type=str, default="false")
        self.add_argument('--no_per_class_acc', action='store_true')
        self.add_argument('--sweep_name', type=str, default="no_sweep")
        self.add_argument('--values_init', type=str, default="zeros", help='rand or zeros')
        self.add_argument('--backbone', type=str, default="dino_resnet50",
                          help="dino_resnet50 or dino_vits8")
        self.add_argument('--t_mode', type=str, default="uniform_importance",
                          help='uniform_importance only mode so far')
        self.add_argument('--scaling_mode', type=str, default="free_num_keys",
                          help='free_num_keys only so far')
        self.add_argument('--batch_size', type=int, default=256,
                          help='Number of tasks in a mini-batch of tasks (default: 16).')
        self.add_argument('--epochs', type=int, default=200,
                          help='epochs to train.')
        self.add_argument('--train_epochs', type=int, default=100,
                          help='epochs to train.')
        self.add_argument('--eval_every_steps', type=int, default=1000,
                          help='log validation every N epochs')
        self.add_argument('--log_step_size', type=int, default=100)
        self.add_argument('--num-workers', type=int, default=8,
                          help='Number of workers for data loading (default: 8).')
        self.add_argument('--seed', type=int, default=42,
                          help='random seed')
        self.add_argument('--dataset_name', type=str, default="CIFAR10",
                          help='CI dataset name')
        self.add_argument('--training_mode', type=str, default="ood",
                          help='iid or ood (class incremental)')
        self.add_argument('--save_checkpoints', type=str, default="false")
        self.add_argument('--add_distance_to_values', type=str, default="false")
        self.add_argument('--accept_image_fmap', type=str, default="false")
        self.add_argument('--init_mode', type=str, default="random", help='kmeans or random')
        self.add_argument('--frozen_encoder', type=str, default="true")
        self.add_argument('--pretrain_data', type=str, default="same",
                          help='pretrain_data')
        self.add_argument('--pretrain_layer', type=int, default=4,
                          help='pretrained_encoder_layer')
        self.add_argument('--cl_epochs', type=int, default=2000,
                          help='cl_epochs')
        self.add_argument('--num_books', type=int, default=1,
                          help='discrete codes num books')
        self.add_argument('--dim_key', type=int, default=14,
                          help='dim_key')
        self.add_argument('--topk', type=int, default=1,
                          help='topk fetched key value pairs')
        self.add_argument('--dim_value', type=int, default=10,
                          help='if dim_value is 0, then it is the same as dim_key')
        self.add_argument('--num_pairs', type=int, default=800,
                          help='num_pairs')
        self.add_argument('--init_epochs', type=int, default=1,
                          help='init_epochs')
        self.add_argument('--split_size', type=int, default=2,
                          help='split_size')
        self.add_argument('--sample_codebook_temperature', type=float, default=0.0)
        self.add_argument('--gradient_clip', type=float, default=1.0)
        self.add_argument('--learning_rate', type=float, default=3e-4,
                          help='learning_rate')
        self.add_argument('--label_smoothing', type=float, default=0.1,
                          help='label_smoothing')
        self.add_argument('--weight_decay', type=float, default=0.0,
                          help='weight_decay')
        self.add_argument('--decay', type=float, default=0.95,
                          help='decay')
        self.add_argument('--threshold_factor', type=float, default=0.1,
                          help='threshold_factor')
        self.add_argument('--ff_dropout', type=float, default=0.0,
                          help='ff_dropout')
        self.add_argument('--no_transforms', type=str, default="false",
                          help='no_transforms')
        self.add_argument('--splitting_mode', type=str, default="random_projection",
                          help='random_projection or learned_projection or chunk')
        self.add_argument('--decoder_model', type=str, default="codebook-voting-logits",
                          help='decoder_model: codebook-voting-logits or linear-probe or mlp-128 or lp-no-bias')
        self.add_argument('--optimizer', type=str, default="SGD",
                          help='Adam or SGD')
        self.add_argument('--method', type=str, default="ours",
                          help='ours or mlp')

    def parse(self):
        args = super().parse_args()
        for key in args.__dict__:
            value = args.__dict__[key]
            if value in ["true", "False", "True", "false"]:
                if value == "true" or value == "True":
                    args.__dict__[key] = True
                else:
                    args.__dict__[key] = False
        args.identifier = str(uuid.uuid4())
        args.root_dir = os.environ.get("PROJECT_ROOT_DIR", "./")
        return args
