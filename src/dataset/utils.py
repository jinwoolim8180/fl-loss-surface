import torch
from argparse import Namespace
from . import cifar10, mnist, femnist, colored_mnist, marked_mnist
import numpy as np

def get_dataset(args, split):
    if args.dataset == 'cifar10':
        dataset = cifar10.get_dataset(split)
    elif args.dataset == 'mnist':
        dataset = mnist.get_dataset(split)
    elif args.dataset == 'femnist':
        dataset = femnist.get_dataset(split)
    elif args.dataset == 'colored_mnist':
        dataset = colored_mnist.get_dataset(split)
    elif args.dataset == 'marked_mnist':
        dataset = marked_mnist.get_dataset(split)
    else:
        raise NotImplementedError('dataset not implemented.')

    if not torch.is_tensor(dataset.targets):
        dataset.targets = torch.tensor(dataset.targets, dtype=torch.long)

    if split == 'train':
        labels = dataset.targets
        label_indices = {l: (labels == l).nonzero().squeeze().type(torch.LongTensor) for l in torch.unique(labels)}
        total_indices = []
        for l, indices in label_indices.items():
            if l < args.n_minority_classes:
                total_indices.append(indices[torch.randperm(len(indices) // args.rho)])
            else:
                total_indices.append(indices)

        total_indices = torch.cat(total_indices)
        dataset.data = dataset.data[total_indices]
        dataset.targets = labels[total_indices]
    return dataset


def split_client_indices(dataset, args: Namespace, coloured=False) -> list:
    if args.dataset == 'femnist':
        return sampling_femnist(dataset, args.clients)
    elif args.dataset == 'colored_mnist' or args.dataset == 'marked_mnist':
        return sampling_colored(dataset, args.clients)
    if args.distribution == 'iid':
        return sampling_iid(dataset, args.clients, coloured=coloured)
    if args.distribution == 'imbalance':
        return sampling_imbalance(dataset, args.clients, args.beta, coloured=coloured)
    if args.distribution == 'dirichlet':
        return sampling_dirichlet(dataset, args.clients, args.beta, coloured=coloured)

def sampling_femnist(dataset: femnist.FEMNISTDataset, num_clients):
    writers = dataset.writers
    return [(writers == w).nonzero().squeeze().type(torch.LongTensor) for w in torch.unique(writers)]


def sampling_colored(dataset, num_clients) -> list:
    client_indices = []
    labels = torch.floor(torch.log10(dataset.targets))
    labels[labels < 0] = 0
    rgb_ratio = [0.33, 0.33, 0.34]
    rgb_client = [int(0.33 * num_clients), int(0.33 * num_clients), num_clients - 2 * int(0.33 * num_clients)]
    imbalanced_indices = []
    # split balanced
    label_indices = [(labels == l).nonzero().squeeze().type(torch.LongTensor) for l in torch.unique(labels)]
    for i, indices in enumerate(label_indices):
        indices = indices[torch.randperm(len(indices))]
        splitted_indices = torch.tensor_split(indices, rgb_client[i])
        client_indices += splitted_indices
    return client_indices


def sampling_iid(dataset, num_clients, coloured=False) -> list:
    client_indices = [torch.tensor([]) for _ in range(num_clients)]
    if coloured:
        labels = torch.floor(torch.log10(dataset.targets))
    else:
        labels = dataset.targets
    for indices in [(labels == l).nonzero().squeeze().type(torch.LongTensor) for l in torch.unique(labels)]:
        indices = indices[torch.randperm(len(indices))]
        splitted_indices = torch.tensor_split(indices, num_clients)
        client_indices = [torch.cat((c_i, s_i)).type(torch.LongTensor) for c_i, s_i in zip(client_indices, splitted_indices)]
    return client_indices


def sampling_imbalance(dataset, num_clients, beta, coloured=False) -> list:
    client_indices = [torch.tensor([]) for _ in range(num_clients)]
    if coloured:
        labels = torch.floor(torch.log10(dataset.targets))
        labels[labels < 0] = 0
    else:
        labels = dataset.targets
    print(labels)
    imbalanced_indices = []
    # split balanced
    label_indices = [(labels == l).nonzero().squeeze().type(torch.LongTensor) for l in torch.unique(labels)]
    for indices in label_indices:
        indices = indices[torch.randperm(len(indices))]
        balanced, imbalanced = torch.tensor_split(indices, [int(len(indices)*beta)])
        imbalanced_indices.append(imbalanced)
        splitted_indices = torch.tensor_split(balanced, num_clients)
        client_indices = [torch.cat((c_i, s_i)).type(torch.LongTensor) for c_i, s_i in zip(client_indices, splitted_indices)]

    # split imbalanced
    imbalanced_indices = torch.cat(imbalanced_indices).type(torch.LongTensor)
    splitted_indices = torch.tensor_split(imbalanced_indices, num_clients)
    client_indices = [torch.cat((c_i, s_i)).type(torch.LongTensor) for c_i, s_i in zip(client_indices, splitted_indices)]
    return client_indices
            

def sampling_dirichlet(dataset, num_clients, beta, coloured=False) -> list:
    min_size = 0
    if coloured:
        labels = torch.floor(torch.log10(dataset.targets))
    else:
        labels = dataset.targets
    while min_size < 10:
        client_indices = [torch.tensor([]) for _ in range(num_clients)]
        for indices in [(labels == l).nonzero().squeeze() for l in torch.unique(labels)]:
            indices = indices[torch.randperm(len(indices))]
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = np.array([p*(len(c_i) < (len(labels) / num_clients)) for p, c_i in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = torch.tensor((np.cumsum(proportions)*len(indices)).astype(int)[:-1])
            splitted_indices = torch.tensor_split(indices, proportions)
            client_indices = [torch.cat((c_i, s_i)).type(torch.LongTensor) for c_i, s_i in zip(client_indices, splitted_indices)]
        min_size = min([len(indices) for indices in client_indices])
    return client_indices