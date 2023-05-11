import logging
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import torchvision
from torchvision.transforms import Compose
import torch


def create_embedding_dataset(
    dataloader_train, dataloader_test, bottlenecked_encoder, args
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset, dataloader in [("train", dataloader_train), ("test", dataloader_test)]:
        z_dataset, y_dataset = None, None
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            with torch.no_grad():
                z = bottlenecked_encoder(x)[-1]
            if i == 0:
                z_dataset = z
                y_dataset = y
            else:
                z_dataset = torch.cat((z_dataset, z), dim=0)
                y_dataset = torch.cat((y_dataset, y), dim=0)
        embedding_file = (
            f"{args.backbone}_{args.pretrain_layer}_{dataset}_embeddings.pt"
        )
        labels_file = f"{args.backbone}_{args.pretrain_layer}_{dataset}_labels.pt"
        # create embeddings directory if it doesn't exist
        if not os.path.exists(os.path.join(args.root_dir, "backbone_embeddings")):
            os.makedirs(os.path.join(args.root_dir, "backbone_embeddings"))
        embedding_path = os.path.join(args.root_dir, "backbone_embeddings", embedding_file)
        labels_path = os.path.join(args.root_dir, "backbone_embeddings", labels_file)
        torch.save(z_dataset.cpu(), embedding_path)
        torch.save(y_dataset.cpu(), labels_path)


class EmbeddingDataset(Dataset):
    def __init__(self, args, dataset: str, train: bool):
        mode = "train" if train else "test"
        embedding_file = (
            f"{args.backbone}_{args.pretrain_layer}_{dataset}_{mode}_embeddings.pt"
        )
        labels_file = (
            f"{args.backbone}_{args.pretrain_layer}_{dataset}_{mode}_labels.pt"
        )
        embedding_path = os.path.join(
            args.root_dir, "backbone_embeddings", embedding_file
        )
        labels_path = os.path.join(args.root_dir, "backbone_embeddings", labels_file)
        if os.path.exists(embedding_path) and os.path.exists(labels_path):
            self.embeddings = torch.load(embedding_path)
            self.labels = torch.load(labels_path)
        else:
            raise FileNotFoundError(
                f"Embedding file {embedding_path} or labels file {labels_path} not found!"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = self.embeddings.to(device)
        self.labels = self.labels.to(device)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        embedding, label = self.embeddings[idx], self.labels[idx]
        return embedding, label


def get_split_dataloaders(dataset, class_list, args):
    if embedding_dataset_exists(dataset, args):
        logging.info(f"Loading split {class_list} from disk")
        dataset_full = EmbeddingDataset(args, dataset, train=True)
        dataset_split = ClassSplittedDataset(dataset=dataset_full,
                                             classes=class_list)
        dataloader = DataLoader(
            dataset=dataset_split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
    elif dataset == "CIFAR10":
        transforms_c10 = Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        cifar10_dataset_train = torchvision.datasets.CIFAR10(
            root=args.root_dir, train=True, download=True, transform=transforms_c10
        )
        dataset_split = ClassSplittedDataset(
            dataset=cifar10_dataset_train, classes=class_list
        )
        dataloader = DataLoader(
            dataset=dataset_split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif dataset == "CIFAR100":
        transforms_c100 = Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        cifar100_dataset_train = torchvision.datasets.CIFAR100(
            root=args.root_dir, train=True, download=True, transform=transforms_c100
        )
        dataset_split = ClassSplittedDataset(
            dataset=cifar100_dataset_train, classes=class_list
        )
        dataloader = DataLoader(
            dataset=dataset_split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        raise NotImplementedError

    return dataloader


def get_embedding_dataloader(dataset, args):
    dataset_train = EmbeddingDataset(args, dataset, train=True)
    dataset_test = EmbeddingDataset(args, dataset, train=False)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    return dataloader_train, dataloader_test


def embedding_dataset_exists(dataset, args):
    embedding_train_file = (
        f"{args.backbone}_{args.pretrain_layer}_{dataset}_train_embeddings.pt"
    )
    labels_train_file = (
        f"{args.backbone}_{args.pretrain_layer}_{dataset}_train_labels.pt"
    )
    embedding_train_path = os.path.join(
        args.root_dir, "backbone_embeddings", embedding_train_file
    )
    labels_train_path = os.path.join(
        args.root_dir, "backbone_embeddings", labels_train_file
    )

    embedding_test_file = (
        f"{args.backbone}_{args.pretrain_layer}_{dataset}_test_embeddings.pt"
    )
    labels_test_file = (
        f"{args.backbone}_{args.pretrain_layer}_{dataset}_test_labels.pt"
    )
    embedding_test_path = os.path.join(
        args.root_dir, "backbone_embeddings", embedding_test_file
    )
    labels_test_path = os.path.join(
        args.root_dir, "backbone_embeddings", labels_test_file
    )

    # check if all paths exist
    if (
        not os.path.exists(embedding_train_path)
        or not os.path.exists(labels_train_path)
        or not os.path.exists(embedding_test_path)
        or not os.path.exists(labels_test_path)
    ):
        return False

    return True


def get_dataloaders(dataset, args):
    if embedding_dataset_exists(dataset, args):
        logging.info("Loading embeddings from disk")
        dataloader_train, dataloader_test = get_embedding_dataloader(dataset, args)
    elif dataset == "CIFAR10":
        transforms_c10 = Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        cifar10_dataset_train = torchvision.datasets.CIFAR10(
            root=args.root_dir, train=True, download=True, transform=transforms_c10
        )
        cifar10_dataset_test = torchvision.datasets.CIFAR10(
            root=args.root_dir, train=False, download=True, transform=transforms_c10
        )
        dataloader_train = DataLoader(
            dataset=cifar10_dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dataloader_test = DataLoader(
            dataset=cifar10_dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif dataset == "STL10":
        transforms_c10 = Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
                transforms.Resize((32, 32)),
            ]
        )
        stl10_dataset_train = torchvision.datasets.STL10(
            root=args.root_dir,
            split="unlabeled",
            download=True,
            transform=transforms_c10,
        )
        stl10_dataset_test = torchvision.datasets.STL10(
            root=args.root_dir, split="test", download=True, transform=transforms_c10
        )
        dataloader_train = DataLoader(
            dataset=stl10_dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dataloader_test = DataLoader(
            dataset=stl10_dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif dataset == "CIFAR100":
        transforms_c100 = Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        cifar100_dataset_train = torchvision.datasets.CIFAR100(
            root=args.root_dir, train=True, download=True, transform=transforms_c100
        )
        cifar100_dataset_test = torchvision.datasets.CIFAR100(
            root=args.root_dir, train=False, download=True, transform=transforms_c100
        )
        dataloader_train = DataLoader(
            dataset=cifar100_dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dataloader_test = DataLoader(
            dataset=cifar100_dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        raise NotImplementedError

    return dataloader_train, dataloader_test


class ClassSplittedDataset(Dataset):
    def __init__(self, dataset, classes, class_to_indices: dict = None):
        self.dataset = dataset
        self.classes = classes
        if class_to_indices is None:
            self.class_to_indices = self._get_class_to_indices()
        else:
            self.class_to_indices = class_to_indices

        self.indices = []
        for c in self.classes:
            self.indices += self.class_to_indices[c]

    def _get_class_to_indices(self):
        class_to_indices = {}
        for i in tqdm(range(len(self.dataset))):
            label = int(self.dataset[i][1])
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(i)
        return class_to_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

