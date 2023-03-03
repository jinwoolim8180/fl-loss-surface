from argparse import Namespace
from modulefinder import Module
import random
import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

import models

class Server:
    def __init__(self, args, train_dataset, indices):
        self.args: Namespace = args
        self.net = models.get_model(args)
        self.weight_sum = 0
        self.model_parameters: Dict[str, torch.Tensor] = models.get_model(args).state_dict()
        self.delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
        self.lambda_var = torch.zeros(self.args.clients)
        self.client_delta = {}
        self.client_losses = [0 for _ in range(self.args.clients)]
        self.client_loaders = []
        for i in range(self.args.clients):
            subset = Subset(train_dataset, indices[i])
            train_loader = DataLoader(subset, self.args.batch_size, shuffle=True, num_workers=0)
            self.client_loaders.append(train_loader)


    def update_client_param(self, client_id, client_delta, weight, loss):
        self.weight_sum += weight
        self.client_losses[client_id] = loss
        self.client_delta[client_id] = client_delta

    def aggregate(self):
        plt.clf()
        plt.hist(self.client_losses, bins=50)
        plt.savefig('loss.png')

        client_L = []
        criterion = nn.CrossEntropyLoss()
        self.net = self.net.cuda()
        self.net.load_state_dict(self.model_parameters)
        for client_id in range(len(self.client_delta)):
            test_loss = 0.
            with torch.no_grad():
                self.net.eval()
                for inputs, labels in self.client_loaders[client_id]:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = self.net(inputs)
                    test_loss += criterion(outputs, labels)
            test_loss /= len(self.client_loaders[client_id])
            dist = 0
            for k in self.delta_avg.keys():
                dist += (self.client_delta[client_id][k] - self.delta_avg[k]).double().norm(2).item()
            client_L.append((test_loss.cpu() - self.client_losses[client_id]) / dist)
        plt.clf()
        plt.hist(client_L)
        plt.savefig('loss_dist.png')
        print("2 figure saved.")

        # clients = self.args.clients
        clients = 10
        weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / clients
        # weights = [x / sum(client_L) for x in client_L]
        self.delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
        client_losses = np.array(self.client_losses)
        topk = np.argpartition(client_losses, -clients)[-clients:]
        # topk = np.random.permutation(self.args.clients)[:clients]
        for client_id in topk:
            for k in self.delta_avg.keys():
                self.delta_avg[k] = self.delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(self.delta_avg[k].dtype)

        for k in self.model_parameters.keys():
            self.model_parameters[k].add_(self.delta_avg[k])

        if self.args.climb:
            client_losses = torch.tensor(self.client_losses)
            self.lambda_var += self.args.lambda_lr * (client_losses - torch.mean(client_losses) - self.args.epsilon) / self.args.clients
            self.lambda_var = torch.clamp(self.lambda_var, min=0., max=100.)

        self.client_delta = {}
        self.weight_sum = 0