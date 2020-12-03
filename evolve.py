import numpy as np
import random
import torch
from network import ConvNet
from eval_nn import eval_nn


def copynet(net):
    newnet = ConvNet(channels=64, blocks=3)
    newnet.load_state_dict(net.state_dict())
    return newnet


class NetworkPopulation:
    def __init__(self, networks, eval_num=100, device='cpu',
                 keep_ratio=5, recombine_prob=0.2, mutation_rate=0.01):
        if isinstance(networks, list):
            self.networks = np.array(networks)
        else:
            self.networks = networks
        self.popsize = len(networks)
        self.keep_ratio = keep_ratio
        self.recombine_prob = recombine_prob
        self.mutation_rate = mutation_rate
        assert self.popsize >= keep_ratio
        self.eval_num = eval_num
        self.device = device
        for m in self.networks:
            m.to(device)
        self.evals = np.zeros(self.popsize)

    def eval(self):
        for i, model in enumerate(self.networks):
            print(f'{i+1} of {self.popsize}')
            mean_log_score, max_score, mean_moves = \
                eval_nn(model, number=self.eval_num, device=self.device)
            print(mean_log_score, max_score, mean_moves)
            self.evals[i] = mean_log_score

    def select(self):
        ind = np.argsort(-1*self.evals)
        self.networks = self.networks[ind]
        self.evals = self.evals[ind]
        top10 = self.networks[:(self.popsize // self.keep_ratio)]
        # TODO: Do kept networks need to be re-evaluated or not?

        remain = []
        for r in np.random.rand(self.popsize - len(top10)):
            if r < self.recombine_prob:
                remain.append(self.recombine(*np.random.choice(top10, 2, replace=False)))
            else:
                remain.append(self.mutate(np.random.choice(top10, 1).item()))

        return np.concatenate([top10, remain])

    def mutate(self, network):
        network = copynet(network)
        for param in network.parameters():
            param.data += (torch.rand_like(param) < self.mutation_rate) * torch.randn_like(param)
        return network

    def recombine(self, network1, network2):
        network1 = copynet(network1)
        network2 = copynet(network2)
        for p1, p2 in zip(network1.parameters(), network2.parameters()):
            if random.getrandbits(1):
                p1.data = p2.data
        return network1

