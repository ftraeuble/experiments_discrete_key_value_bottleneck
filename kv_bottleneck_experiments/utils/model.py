import argparse

import einops
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

import kv_bottleneck_experiments.utils.base as base_utils


class CodebookVotingLogitsDecoder(nn.Module):
    def __init__(self, dim_values, class_nums, args: argparse.Namespace):
        super().__init__()
        self.dim_values = dim_values
        self.class_nums = class_nums
        self.args = args

        if dim_values != class_nums:
            raise ValueError("dim_values must be equal to class_nums")

        weights = torch.ones(args.topk, dtype=torch.float32)
        self.register_buffer("weights", weights)

        self.dropout_layer = nn.Dropout1d(p=float(self.args.ff_dropout))

    def forward(self, x):
        x = einops.rearrange(x, "b c t v ... -> b c t (...) v")
        x = einops.rearrange(x, "b c t ... v -> b (... c) t v")
        x = torch.einsum("b n t v, t -> b n v", x, self.weights)
        x = self.dropout_layer(x)
        x = einops.reduce(x, "b n v -> b v", "mean")
        return x


def get_encoder_output_shape(args):
    if args.backbone == "dino_resnet50":
        h, num_channels, w = get_resnet50_output_shape(args)
    elif "swav" in args.backbone or "vicreg" in args.backbone:
        if "w2" in args.backbone:
            LAYER_TO_CHANNELS = {1: 2*256, 2: 2*512, 3: 2*1024, 4: 2*2048}
        else:
            LAYER_TO_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}
        if args.accept_image_fmap:
            LAYER_TO_HW = {1: 8, 2: 4, 3: 2, 4: 1}
            h = LAYER_TO_HW[args.pretrain_layer]
            w = LAYER_TO_HW[args.pretrain_layer]
        else:
            h = 1
            w = 1
        num_channels = LAYER_TO_CHANNELS[args.pretrain_layer]
    elif args.backbone == "cifar10_resnet56" or args.backbone == "cifar100_resnet56":
        LAYER_TO_CHANNELS = {1: 16, 2: 32, 3: 64}
        if args.accept_image_fmap:
            LAYER_TO_HW = {1: 8, 2: 4, 3: 2, 4: 1}
            h = LAYER_TO_HW[args.pretrain_layer]
            w = LAYER_TO_HW[args.pretrain_layer]
        else:
            h = 1
            w = 1
        num_channels = LAYER_TO_CHANNELS[args.pretrain_layer]
    elif args.backbone == "resnet50_imagenet_v2":
        h, num_channels, w = get_resnet50_output_shape(args)
    elif args.backbone == "dino_vits8":
        num_channels = 384
        h = 1
        w = 1
    elif args.backbone == "convmixer":
        num_channels = 256
        h = 1
        w = 1
    elif args.backbone == "clip_vit_b32":
        num_channels = 512
        h = 1
        w = 1
    else:
        raise NotImplementedError("Backbone {} not supported".format(args.backbone))
    return num_channels, h, w


def get_resnet50_output_shape(args):
    LAYER_TO_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}
    if args.accept_image_fmap:
        LAYER_TO_HW = {1: 8, 2: 4, 3: 2, 4: 1}
        h = LAYER_TO_HW[args.pretrain_layer]
        w = LAYER_TO_HW[args.pretrain_layer]
    else:
        h = 1
        w = 1
    num_channels = LAYER_TO_CHANNELS[args.pretrain_layer]
    return h, num_channels, w


def get_decoder_module(num_codebooks, dim_values, dim_keys, num_channels, args):
    class_nums = base_utils.get_class_nums(args)
    decoder_modules = []

    if args.add_distance_to_values:
        dim_values = dim_values + 1  # append distance to retrieved value

    if args.decoder_model == "linear-probe":
        dim_value_in = num_codebooks * dim_values

        if "mlp" in args.method:
            dim_value_in = num_channels

        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, class_nums),
        ]
    elif args.decoder_model == "mlp-128":
        dim_value_in = num_codebooks * dim_values

        if "mlp" in args.method:
            dim_value_in = num_channels

        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, 128),
            nn.ReLU(),
            nn.Linear(128, class_nums),
        ]
    elif args.decoder_model == "lp-no-bias":
        dim_value_in = num_codebooks * dim_values

        if "mlp" in args.method:
            dim_value_in = num_channels

        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, class_nums, bias=False),
        ]
    elif args.decoder_model == "codebook-voting-logits":
        decoder = CodebookVotingLogitsDecoder(dim_values, class_nums, args=args)
        return decoder
    else:
        raise NotImplementedError(
            "Decoder size {} not supported".format(args.decoder_model)
        )
    decoder = nn.Sequential(*decoder_modules)
    return decoder


def get_threshold_ema_dead_code(args):
    num_channels, h, w = get_encoder_output_shape(args)
    if args.t_mode == "uniform_importance":
        num_pairs = get_key_value_pairs_per_codebook(args)
        threshold = args.threshold_factor * args.batch_size * h * w / num_pairs
    else:
        raise NotImplementedError(f"args.t_mode = {args.threshold}")
    return threshold


def get_key_value_pairs_per_codebook(args):
    if args.scaling_mode == "constant_num_keys":
        num_channels, h, w = get_encoder_output_shape(args)
        key_value_pairs_per_codebook = round(
            args.num_pairs * num_channels / args.num_books
        )
    elif args.scaling_mode == "free_num_keys":
        key_value_pairs_per_codebook = args.num_pairs
    else:
        raise NotImplementedError(f"Not implemented mode")
    return key_value_pairs_per_codebook


class ModelWrapper(nn.Module):
    def __init__(self, bottlenecked_encoder, decoder, args: argparse.Namespace):
        super(ModelWrapper, self).__init__()
        self.bottlenecked_encoder = bottlenecked_encoder
        self.bottlenecked_encoder.freeze_encoder()
        self.decoder = decoder
        self.args = args
        if self.args.method == "ours":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.args.method == "kv_tune_full_decoder":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.args.method == "mlp":
            self.tuple_pos = -1  # this is the position of the returned key codes
        else:
            raise NotImplementedError("Method {} not supported".format(args.method))
        self.dropout_layer = nn.Dropout1d(p=float(self.args.ff_dropout))

    def forward(self, x):
        if self.args.method == "mlp":
            if not isinstance(self.bottlenecked_encoder.encoder, nn.Identity):
                bottleneck_tuple = self.bottlenecked_encoder(x)[self.tuple_pos]
                x = bottleneck_tuple.clone().detach()
            x = self.decoder(x)
        else:
            bottleneck_tuple = self.bottlenecked_encoder(x)

            x = bottleneck_tuple[self.tuple_pos]
            if self.args.decoder_model == "codebook-voting-logits":
                x = self.decoder(x=x)
                return x
            if len(x.shape) == 4:
                x = einops.rearrange(x, "b c t v -> b (c t) v")
            if len(x.shape) == 5:
                x = einops.rearrange(x, "b c v h w -> b (c h w) v")
            if len(x.shape) == 6:
                x = einops.rearrange(x, "b c t v h w -> b (c t h w) v")
            x = self.dropout_layer(x)
            if self.args.add_distance_to_values:
                distances = bottleneck_tuple[3]
                x = torch.cat((x, distances), dim=2)
            x = self.decoder(x)
        return x

    def forward_decoder(self, x):
        if self.args.method == "mlp":
            x = self.decoder(x)
        else:
            x = self.dropout_layer(x)
            x = self.decoder(x)
        return x

    def freeze_for_adaptation(self):
        if self.args.method == "ours":
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif self.args.method == "kv_tune_full_decoder":
            return
        elif self.args.method == "mlp":
            return
        else:
            raise NotImplementedError(
                "Method {} not supported".format(self.args.method)
            )
