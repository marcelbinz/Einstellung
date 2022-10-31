import data
import torch
from torch.distributions import Normal, Categorical, kl_divergence
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Einstellung')
parser.add_argument('--id', type=int, default=1, help='task id')
args = parser.parse_args()

class Model(nn.Module):
    def __init__(self, num_actions, beta=10, alpha=0.0):
        super(Model, self).__init__()
        self.beta = beta
        self.params = nn.Parameter(torch.zeros(3, num_actions))
        self.optimizer = optim.SGD(self.parameters(), lr=alpha)

    def forward(self, utilities):
        self.probabilities = F.softmax(self.params, dim=1)
        joint_prior = torch.einsum('i,j,k->ijk', self.probabilities[0], self.probabilities[1], self.probabilities[2]).flatten()

        posterior = joint_prior * torch.exp(self.beta * utilities)
        posterior = posterior / posterior.sum()

        return Categorical(posterior).sample(), kl_divergence(Categorical(posterior), Categorical(joint_prior)).item()

    def learn(self, action):
        prior = Categorical(self.probabilities)
        loss = -prior.log_prob(combinations[action] + 10).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class RationalModel(nn.Module):
    def __init__(self):
        super(RationalModel, self).__init__()

    def forward(self, utilities):
        return torch.argmax(utilities), 0

    def learn(self, action):
        pass

experiments = [data.default, data.control, data.less_e, data.more_e, data.alternating, data.extreme]
num_runs = 100

# only search for weights between -10 and +10
values = torch.arange(-10, 11, 1)
combinations = torch.cartesian_prod(values, values, values)

def simulate(experiments, num_runs, gamma, bounded, alphas, betas, use_l1=False):
    for num_experiment, experiment in enumerate(experiments):
        x, y = experiment()

        # precompute utilities
        optimality = ((combinations.float() @ x.t()) == y).double()

        if use_l1:
            utilities = optimality - gamma * combinations.abs().sum(-1, keepdims=True)
        else:
            utilities = optimality - gamma * (combinations ** 2).sum(-1, keepdims=True)

        num_correct = torch.zeros(num_runs, len(alphas), len(betas), x.shape[0])
        num_d_solution = torch.zeros(num_runs, len(alphas), len(betas), x.shape[0])
        num_e_solution = torch.zeros(num_runs, len(alphas), len(betas), x.shape[0])
        klds = torch.zeros(num_runs, len(alphas), len(betas), x.shape[0])

        for alpha_idx, alpha in enumerate(alphas):
            print(alpha_idx)
            for beta_idx, beta in enumerate(betas):
                for run in range(num_runs):
                    if bounded:
                        model = Model(values.shape[0], beta, alpha)
                    else:
                        model = RationalModel()
                    for i in range(x.shape[0]):
                        # inference
                        action, kld = model(utilities[:, i])
                        klds[run, alpha_idx, beta_idx, i] = kld
                        # adjust prior
                        model.learn(action)

                        if optimality[action, i].bool().item():
                            num_correct[run, alpha_idx, beta_idx, i] +=1

                        if optimality[action, i].bool().item() and combinations[action].pow(2).sum() < 6:
                            num_d_solution[run, alpha_idx, beta_idx, i] +=1

                        if optimality[action, i].bool().item() and combinations[action].pow(2).sum() == 6:
                            num_e_solution[run, alpha_idx, beta_idx, i] +=1
        if use_l1:
            torch.save([num_correct, num_d_solution, num_e_solution, klds], 'data/l1_exp' + str(num_experiment) + '_gamma_' + str(gamma) + '_bounded_' + str(bounded) + '_run_' + str(args.id) + '.pth')
        else:
            torch.save([num_correct, num_d_solution, num_e_solution, klds], 'data/exp' + str(num_experiment) + '_gamma_' + str(gamma) + '_bounded_' + str(bounded) + '_run_' + str(args.id) + '.pth')

# simulate L1
gamma = 0.05
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas, use_l1=True)

# simulate full
gamma = 0.05
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)

# simulate no physical
gamma = 0.00
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)

# simulate rational
gamma = 0.05
bounded = False

alphas = torch.linspace(1, 1, 1)
betas = torch.linspace(1, 1, 1)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)

# simulate other gammas
gamma = 0.025
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)

gamma = 0.1
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)

gamma = 0.2
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)

gamma = 0.4
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)

gamma = 0.8
bounded = True

alphas = torch.linspace(0., 1, 21)
betas = torch.linspace(1, 50, 50)

simulate(experiments, num_runs, gamma, bounded, alphas, betas)
