import logging
import os
from pathlib import Path

import torch
import wandb
import matplotlib as mpl
from torch import nn
import key_value_bottleneck.core as kv_core
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

import kv_bottleneck_experiments.utils.argparse as argparse_utils
import kv_bottleneck_experiments.utils.base as base_utils
import kv_bottleneck_experiments.utils.model as model_utils
import kv_bottleneck_experiments.utils.eval as eval_utils
import kv_bottleneck_experiments.utils.train as train_utils
import kv_bottleneck_experiments.utils.data as data_utils


PROJECT_NAME = os.environ.get("PROJECT_NAME", "PLACEHOLDER_PROJECT_NAME")
PROJECT_ENTITY = os.environ.get("PROJECT_ENTITY", "PLACEHOLDER_PROJECT_ENTITY")


def initialize_model(dataloader_train, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device variable: " + str(device))

    key_value_pairs_per_codebook = model_utils.get_key_value_pairs_per_codebook(args)
    threshold_ema_dead_code = model_utils.get_threshold_ema_dead_code(args)

    if "cifar10" in args.backbone or "cifar100" in args.backbone:
        bottleneck_encoder_cls = kv_core.SLPretrainedBottleneckedEncoder
    elif args.backbone == "resnet50_imagenet_v2":
        bottleneck_encoder_cls = kv_core.SLPretrainedBottleneckedEncoder
    elif args.backbone == "clip_vit_b32":
        bottleneck_encoder_cls = kv_core.CLIPBottleneckedEncoder
    elif "vicreg" in args.backbone:
        bottleneck_encoder_cls = kv_core.VICRegBottleneckedEncoder
    elif "swav" in args.backbone:
        bottleneck_encoder_cls = kv_core.SwavBottleneckedEncoder
    elif "convmixer" in args.backbone:
        bottleneck_encoder_cls = kv_core.ConvMixerBottleneckedEncoder
    else:
        bottleneck_encoder_cls = kv_core.DinoBottleneckedEncoder

    bottlenecked_encoder = bottleneck_encoder_cls(
        num_codebooks=args.num_books,
        key_value_pairs_per_codebook=key_value_pairs_per_codebook,
        backbone=args.backbone,
        extracted_layer=args.pretrain_layer,
        pool_embedding=not args.accept_image_fmap,
        init_mode=args.init_mode,
        splitting_mode=args.splitting_mode,
        dim_values=args.dim_value,
        dim_keys=args.dim_key,
        decay=args.decay,
        eps=1e-5,
        threshold_ema_dead_code=threshold_ema_dead_code,
        concat_values_from_all_codebooks=False,
        sample_codebook_temperature=args.sample_codebook_temperature,
        return_values_only=False,
        topk=args.topk,
        input_is_embedding=isinstance(
            dataloader_train.dataset, data_utils.EmbeddingDataset
        ),
    )
    deviating_transforms = None
    if bottlenecked_encoder.transforms is not None:
        deviating_transforms = bottlenecked_encoder.transforms
        if isinstance(dataloader_train, torch.utils.data.DataLoader):
            dataloader_train.dataset.transform = deviating_transforms

    logging.info("Initializing model")
    bottlenecked_encoder.to(device)
    bottlenecked_encoder = bottlenecked_encoder.prepare(
        loader=dataloader_train, epochs=args.init_epochs
    )

    bottlenecked_encoder.reset_cluster_size_counter()

    bottlenecked_encoder.disable_update_keys()
    decoder = model_utils.get_decoder_module(
        num_codebooks=bottlenecked_encoder.num_codebooks,
        dim_values=bottlenecked_encoder.dim_values,
        dim_keys=bottlenecked_encoder.dim_keys,
        num_channels=bottlenecked_encoder.num_channels,
        args=args,
    )
    model = model_utils.ModelWrapper(bottlenecked_encoder, decoder, args)
    model.to(device)
    model.train()

    return model, deviating_transforms


def train(args):
    base_utils.seed_everything(args.seed)
    logging.info("Training run configs: " + str(args))
    dataloader_train, dataloader_test = data_utils.get_dataloaders(
        dataset=args.dataset_name, args=args
    )

    if args.pretrain_data == args.dataset_name:
        model, deviating_transforms = initialize_model(dataloader_train, args)
    else:
        dataloader_pretrain, _ = data_utils.get_dataloaders(
            dataset=args.pretrain_data, args=args
        )
        model, deviating_transforms = initialize_model(dataloader_pretrain, args)

    if deviating_transforms is not None:
        dataloader_train.dataset.transform = deviating_transforms
        dataloader_test.dataset.transform = deviating_transforms

    if args.values_init == "randn":
        values = torch.randn_like(model.bottlenecked_encoder.bottleneck.values)
    elif args.values_init == "rand":
        values = torch.rand_like(model.bottlenecked_encoder.bottleneck.values)
    elif args.values_init == "zeros":
        values = torch.zeros_like(model.bottlenecked_encoder.bottleneck.values)
    else:
        raise ValueError("Unknown values_init: " + args.values_init)
    model.bottlenecked_encoder.bottleneck.values = nn.Parameter(values)
    model.bottlenecked_encoder.bottleneck.values.requires_grad = True

    if args.training_mode == "ood":
        epoch_factor = 1
        if args.seed not in [0, 42]:
            class_split = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            import random

            random.shuffle(class_split)
            class_splits = []
            split_size = args.split_size
            assert split_size in [1, 2, 5, 10]
            num_splits = 10 // split_size
            epoch_factor = split_size / 2
            for i in range(num_splits):
                class_splits.append(class_split[i * split_size: (i + 1) * split_size])
            logging.info("class_splits: " + str(class_splits))
        else:
            class_splits = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    elif args.training_mode == "iid":
        epoch_factor = 5
        class_splits = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    else:
        raise ValueError("Unknown train_mode: " + args.training_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = train_utils.get_optimizer(model, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    model.bottlenecked_encoder.freeze_keys()
    model.freeze_for_adaptation()
    logging.info("model is on device: " + str(next(model.parameters()).device))
    logging.info("device variable: " + str(device))
    log_step_size = args.log_step_size
    loss_total = 0.0
    train_epochs = 0
    epoch = 0
    eval_train_test_accuracy(
        args, criterion, dataloader_test, dataloader_train, model, train_epochs, epoch
    )
    model.train()
    model.bottlenecked_encoder.activate_counts()
    if args.save_checkpoints:
        model_dir = os.path.join(args.root_dir, "checkpoints", args.sweep_name)
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model,
            os.path.join(model_dir, f"{wandb.run.name}_epoch_{str(train_epochs)}.pt"),
        )
    for class_list in class_splits:
        dataloader = data_utils.get_split_dataloaders(
            args.dataset_name, class_list, args
        )
        if deviating_transforms is not None:
            dataloader.dataset.dataset.transform = deviating_transforms
        for epoch in range(0, args.cl_epochs):
            train_epochs += epoch_factor
            if epoch % log_step_size == 1:
                wandb.log({"local_loss": loss_total})
                wandb.log(
                    {
                        "norm_values": torch.norm(
                            model.bottlenecked_encoder.bottleneck.values, dim=-1
                        ).mean()
                    }
                )
                eval_train_test_accuracy(
                    args, criterion, dataloader_test, dataloader_train, model, train_epochs, epoch
                )
                model.train()
                model.bottlenecked_encoder.activate_counts()
                logging.info("cl epoch: " + str(epoch))
            loss_total = 0.0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                loss_total += float(loss.item())
        # eval at very end
        wandb.log({"local_loss": loss_total})
        wandb.log(
            {
                "norm_values": torch.norm(
                    model.bottlenecked_encoder.bottleneck.values, dim=-1
                ).mean()
            }
        )
        eval_train_test_accuracy(
            args, criterion, dataloader_test, dataloader_train, model, train_epochs, epoch
        )
        plot_dict = {}
        for num_book in range(
            min(1, model.bottlenecked_encoder.bottleneck.num_codebooks)
        ):
            plot_dict[f"kv_pair_counts_book_{num_book}"] = wandb.Histogram(
                model.bottlenecked_encoder.bottleneck.cluster_size_counter.detach().cpu()
            )
            if args.plot_pair_utilization:
                values_im = plt.imshow(
                    model.bottlenecked_encoder.bottleneck.values[num_book, :100]
                    .detach()
                    .cpu(),
                    aspect="auto",
                    cmap=mpl.colormaps["hot"],
                    vmax=1,
                    vmin=-1,
                )
                plot_dict[f"values_book_{num_book}"] = wandb.Image(values_im)
        if args.plot_pair_utilization:
            counter_im = plt.imshow(
                model.bottlenecked_encoder.bottleneck.cluster_size_counter[:, :100]
                .detach()
                .cpu(),
                aspect="auto",
                cmap=mpl.colormaps["hot"],
            )
            plot_dict["pair_counts"] = wandb.Image(counter_im)
            classes_image = plt.imshow(
                model.bottlenecked_encoder.bottleneck.values.argmax(dim=-1)[:, :100]
                .detach()
                .cpu(),
                aspect="auto",
                cmap=mpl.colormaps["tab10"],
            )
            wandb.log({f"class_preds": [wandb.Image(classes_image)]})
            plot_dict["class_preds"] = wandb.Image(classes_image)
        plot_dict["fraction_unused_keys"] = float(
            model.bottlenecked_encoder.fraction_of_unused_keys()
        )
        wandb.log(plot_dict)
        if args.save_checkpoints:
            model_dir = os.path.join(args.root_dir, "checkpoints", args.sweep_name)
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model,
                os.path.join(
                    model_dir, f"{wandb.run.name}_epoch_{str(train_epochs)}.pt"
                ),
            )


def eval_train_test_accuracy(
    args, criterion, dataloader_test, dataloader_train, model, train_epochs, epoch
):
    model.eval()
    model.bottlenecked_encoder.deactivate_counts()
    _ = eval_utils.evaluate_accuracy(
        model=model,
        dataloader=dataloader_test,
        train_step=train_epochs,
        epoch=epoch,
        args=args,
        dataset="test",
        wandb_log=True,
        criterion=criterion,
    )
    _ = eval_utils.evaluate_accuracy(
        model=model,
        dataloader=dataloader_train,
        train_step=train_epochs,
        epoch=epoch,
        args=args,
        dataset="train",
        wandb_log=True,
        criterion=criterion,
    )


if __name__ == "__main__":
    run_args = argparse_utils.ArgumentParserWrapper().parse()
    wandb.init(
        project=PROJECT_NAME,
        entity=PROJECT_ENTITY,
        dir=f"{run_args.root_dir}/wandb",
        settings=wandb.Settings(_disable_stats=True),
    )
    train(run_args)
